import os
import re
import math
import time
import itertools
import cv2
import numpy as np
import torch
from scipy.stats import truncnorm
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.utils as vutils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import lib.datasets as datasets
from lib.utils.transforms import fliplr_pts, kpt_affine
from lib.utils.evaluation import accuracy, OffsetCycleAverageMeter, parse_map, generate_ans, generate_mpii_ans
from lib.utils.imutils import batch_resize
from lib.models.displacechan import DisplaceChannel, OffsetTransformer
from lib.models.spacenorm import SpaceNormalization
from utils.globals import config, hparams, globalvars
from utils.log import log_i
from utils.train import adjust_learning_rate
from utils.checkpoint import save_pred, load_pretrained_loose, save_checkpoint, RejectLoadError
from utils.miscs import nprand_init
from experiments.baseexperiment import BaseExperiment, EpochContext
from lib.utils.augtrans import transform_maps

FACTOR = 4

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, num_groups=32):
        assert num_features % num_groups == 0, "num_features({}) is not dividend by num_groups({})".format(num_features, num_groups)
        super(GroupNormWrapper, self).__init__(num_groups, num_features, eps=1e-5)

class Experiment(BaseExperiment):
    def init(self):
        super().init()

        if self.offset_optimizer is not None:
            self.move_dis_avgmeter = []
            for dm in globalvars.displace_mods:
                if dm.offset.size(0) == 0:
                    continue
                self.move_dis_avgmeter.append(OffsetCycleAverageMeter(hparams.LOG.MOVE_AVERAGE_CYCLE, (dm.offset.data * dm.offset_scale).cpu()))
            self.change_sigma_avgmeter = []
            for dp in globalvars.dpools:
                self.change_sigma_avgmeter.append(OffsetCycleAverageMeter(hparams.LOG.SIGMA_CHANGE_AVERAGE_CYCLE, dp.sigma.detach().cpu().abs()))
        else:
            self.move_dis_avgmeter = None
            self.change_sigma_avgmeter = None

    def init_dataset(self):
        self.data_source = hparams.DATASET.PROFILE
        if self.data_source == "coco":
            self.num_parts = datasets.mscoco.NUM_PARTS
            self.flip_index = datasets.mscoco.FLIP_INDEX
        elif self.data_source == "mpii":
            self.num_parts = datasets.mpii.NUM_PARTS
            self.flip_index = datasets.mpii.FLIP_INDEX
        else:
            assert False

        if self.data_source == "coco":
            self.init_mscoco()
        elif self.data_source == "mpii":
            self.init_mpii()

        if hparams.DATASET.SUBSET is not None:
            if self.train_dataset:
                self.train_dataset = Subset(self.train_dataset, list(range(int(len(self.train_dataset) * hparams.DATASET.SUBSET))))

    def init_model(self):
        globalvars.displace_mods = list()
        globalvars.dpools = list()

        self.model = nn.DataParallel(MyPose(self.num_parts).cuda())
        if hparams.MODEL.REGRESS_PREDICT:
            self.pose_regressor = RegressPredictor(
                hparams.MODEL.OUT_SHAPE[1],
                hparams.MODEL.OUT_SHAPE[0]).cuda()

    def init_optimizer(self):
        def _print_parameters(para_groups, all_para_pair):
            print("Parameter groups:")
            para2name_dict = dict(map(lambda p: (p[1], p[0]), all_para_pair))
            for group_name, paras in para_groups.items():
                print("  " + group_name + ":")
                for para_name in map(para2name_dict.get, paras):
                    print("    " + para_name)
                print("")

        # Separate parameters
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in globalvars.displace_mods if hasattr(dm, "offset")]))
            offset_parameter_ids = list(map(lambda x: id(x), self.offset_parameters))
            
            self.offset_regressor_parameters = list(filter(lambda x: x.requires_grad, list(itertools.chain.from_iterable([dm.offset_regressor.parameters() for dm in globalvars.displace_mods if hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET]))))

            if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
                self.dpool_parameters = list(filter(lambda x: x.requires_grad, [dp.sigma for dp in globalvars.dpools]))
            else:
                self.dpool_parameters = []
            dpool_parameter_ids = list(map(lambda x: id(x), self.dpool_parameters))

            if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE:
                self.offset_transformer_parameters = list(filter(
                    lambda x: id(x) not in offset_parameter_ids and id(x) not in dpool_parameter_ids,
                    filter(
                        lambda x: x.requires_grad,
                        list(itertools.chain.from_iterable([dm.offset_transformer.parameters() for dm in globalvars.displace_mods if dm.offset_transformer is not None])) + \
                        (list(self.model.module.transformer.parameters()) if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT else []))))
            else:
                self.offset_transformer_parameters = []
        else:
            self.offset_parameters = []
            self.offset_regressor_parameters = []
            self.offset_transformer_parameters = []
            self.dpool_parameters = []

        all_special_parameters = self.offset_parameters + \
            self.offset_regressor_parameters + \
            self.offset_transformer_parameters + \
            self.dpool_parameters

        special_parameter_ids = list(map(lambda x: id(x), all_special_parameters))
        self.general_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))
        
        _print_parameters(
            dict(
                offset=self.offset_parameters,
                offset_regressor=self.offset_regressor_parameters,
                offset_transformer=self.offset_transformer_parameters,
                dpool=self.dpool_parameters),
            self.model.named_parameters())

        # Make sure no parameter is shared
        all_parameter_ptrs = list(map(lambda x: x.data_ptr(), all_special_parameters + self.general_parameters))
        assert len(all_parameter_ptrs) == len(np.unique(all_parameter_ptrs)), "shared parameter exists"

        # Initialize optimizers
        # Normal optimizer
        self.optimizer = torch.optim.Adam(
            self.general_parameters,
            lr=hparams.TRAIN.LEARNING_RATE,
            weight_decay=hparams.TRAIN.WEIGHT_DECAY)

        # Offset and regressor optimizer
        offset_optimizer_args = []
        if len(self.offset_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_lr", "params": self.offset_parameters, "lr": hparams.TRAIN.OFFSET.LR, "init_lr": hparams.TRAIN.OFFSET.LR})
        if len(self.offset_regressor_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_regressor_lr", "params": self.offset_regressor_parameters, "lr": hparams.TRAIN.OFFSET.LR_REGRESSOR, "init_lr": hparams.TRAIN.OFFSET.LR_REGRESSOR})
        if len(self.dpool_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_dpool_lr", "params": self.dpool_parameters, "lr": hparams.TRAIN.OFFSET.LR_DPOOL_SIGMA, "init_lr": hparams.TRAIN.OFFSET.LR_DPOOL_SIGMA})

        if len(offset_optimizer_args) > 0:
            self.offset_optimizer = torch.optim.Adam(offset_optimizer_args)
        else:
            self.offset_optimizer = None

        # Transformer optimizer
        if len(self.offset_transformer_parameters) > 0:
            self.transformer_optimizer = \
                torch.optim.Adam([{"para_name": "offset_regressor_lr", "params": self.offset_transformer_parameters, "lr": hparams.TRAIN.OFFSET.LR_TRANSFORMER, "init_lr": hparams.TRAIN.OFFSET.LR_TRANSFORMER}])
        else:
            self.transformer_optimizer = None

        self.update_weight = True
        self.update_offset = True
        self.update_transformer = True

    def init_dataloader(self):
        self.worker_init_fn = nprand_init
        super().init_dataloader()

    def init_mscoco(self):
        kpmap_res = None if hparams.MODEL.REGRESS_PREDICT else hparams.MODEL.OUT_SHAPE
        self.coco = COCO("data/mscoco/person_keypoints_train2017.json")
        self.train_dataset = datasets.COCOSinglePose(
            "data/mscoco/images2017",
            self.coco,
            "data/mscoco/sp_split_2017.pth",
            "data/mscoco/" + hparams.DATASET["COCO"].MEAN_STD_FILE,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            ext_border=hparams.DATASET["COCO"].EXT_BORDER,
            kpmap_res=kpmap_res,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNEL,
            scale_factor=hparams.DATASET["COCO"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["COCO"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["COCO"].TRANSLATION_FACTOR)

        self.val_dataset = datasets.COCOSinglePose(
            "data/mscoco/images2017",
            self.coco,
            "data/mscoco/sp_split_2017.pth",
            "data/mscoco/" + hparams.DATASET["COCO"].MEAN_STD_FILE,
            False,
            img_res=hparams.MODEL.INP_SHAPE,
            ext_border=hparams.DATASET["COCO"].EXT_BORDER,
            kpmap_res=kpmap_res,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNEL,
            scale_factor=hparams.DATASET["COCO"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["COCO"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["COCO"].TRANSLATION_FACTOR)
        self.train_collate_fn = datasets.COCOSinglePose.collate_function
        self.valid_collate_fn = datasets.COCOSinglePose.collate_function

    def init_mpii(self):
        kpmap_res = None if hparams.MODEL.REGRESS_PREDICT else hparams.MODEL.OUT_SHAPE
        self.train_dataset = datasets.MPII("data/mpii/images",
            "data/mpii/mpii_human_pose.json",
            "data/mpii/split_sig.pth",
            "data/mpii/mean_std.pth",
            True,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            kpmap_res=kpmap_res,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNEL,
            scale_factor=hparams.DATASET["MPII"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["MPII"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["MPII"].TRANSLATION_FACTOR)

        self.val_dataset = datasets.MPII("data/mpii/images",
            "data/mpii/mpii_human_pose.json",
            "data/mpii/split_sig.pth",
            "data/mpii/mean_std.pth",
            False,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            kpmap_res=kpmap_res,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNEL,
            scale_factor=hparams.DATASET["MPII"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["MPII"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["MPII"].TRANSLATION_FACTOR)
        self.train_collate_fn = datasets.MPII.collate_function
        self.valid_collate_fn = datasets.MPII.collate_function

    def load_checkpoint(self, checkpoint_full,
                        no_strict_model_load=False,
                        no_criterion_load=False,
                        no_optimizer_load=False):

        # Load checkpoint data
        checkpoint = torch.load(checkpoint_full)
        if no_strict_model_load:
            model_state_dict = self.model.state_dict()
            try:
                model_state_dict = load_pretrained_loose(model_state_dict, checkpoint["state_dict"])
            except RejectLoadError:
                return None
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        if not no_optimizer_load:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.offset_optimizer:
                self.offset_optimizer.load_state_dict(checkpoint["offset_optimizer"])
            if self.transformer_optimizer:
                self.transformer_optimizer.load_state_dict(checkpoint["transformer_optimizer"])

        self.move_dis_avgmeter = checkpoint["move_dis_avgmeter"]
        self.change_sigma_avgmeter = checkpoint["change_sigma_avgmeter"]
        if hparams.MODEL.REGRESS_PREDICT:
            self.pose_regressor.load_state_dict(checkpoint["pose_regressor"])
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_full, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict() if self.offset_optimizer else None,
            "transformer_optimizer": self.transformer_optimizer.state_dict() if self.transformer_optimizer else None,
            "move_dis_avgmeter": self.move_dis_avgmeter,
            "change_sigma_avgmeter": self.change_sigma_avgmeter
        }
        if hparams.MODEL.REGRESS_PREDICT:
            checkpoint_dict["pose_regressor"] = self.pose_regressor.state_dict()
        save_checkpoint(checkpoint_dict, checkpoint_full=checkpoint_full, force_replace=True)

    def evaluate(self, epoch_ctx:EpochContext, epoch, step):
        if "annotates" not in epoch_ctx.stored:
            return

        try:
            tb_writer = globalvars.main_context.tb_writer
        except AttributeError:
            tb_writer = None

        if self.data_source == "coco":
            annotates = epoch_ctx.stored["annotates"]
            image_ids = annotates["image_index"]
            ans = annotates["annotate"]
            if ans is not None and len(ans) > 0:
                coco_dets = self.coco.loadRes(ans)
                coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
                coco_eval.params.imgIds = list(image_ids)
                coco_eval.params.catIds = [1]
                coco_eval.evaluate()
                coco_eval.accumulate()

                coco_eval.summarize()
                stats = coco_eval.stats
            else:
                print("No points")
                stats = np.zeros((10,))

            if tb_writer:
                stat_typenames = [
                    ("AP", "avg"),
                    ("AP", "i50"),
                    ("AP", "i75"),
                    ("AP", "med"),
                    ("AP", "lar"),
                    ("AR", "avg"),
                    ("AR", "i50"),
                    ("AR", "i75"),
                    ("AR", "med"),
                    ("AR", "lar")
                ]
                for istat, (stat_type, stat_name) in enumerate(stat_typenames):
                    tb_writer.add_scalar("coco/{}_{}".format(stat_type, stat_name), stats[istat], step)

        elif self.data_source == "mpii":
            annotates = epoch_ctx.stored["annotates"]
            acc = accuracy(annotates["pred"], annotates["gt"], annotates["head_box"])
            if tb_writer:
                tb_writer.add_scalar("mpii/{}_{}".format("PCKh", "avg"), float(acc[0]), step)
            results = list()
            results.append("avg: {:2.2f}".format(float(acc[0]) * 100))
            for i in range(0, acc.size(0)-1):
                if tb_writer:
                    tb_writer.add_scalar("mpii/{}_{}".format("PCKh", datasets.mpii.PART_LABELS[i]), float(acc[i+1]), step)
                results.append("{}: {:2.2f}".format(datasets.mpii.PART_LABELS[i], float(acc[i+1]) * 100))
            print(" | ".join(results) + "\n")

    def process_stored(self, epoch_ctx:EpochContext, epoch, step):
        if config.store:
            for store_key in epoch_ctx.stored:
                if epoch == 0:
                    pred_file = "{}_evaluate.npy".format(store_key)
                else:
                    pred_file = "{}_{}.npy".format(store_key, epoch)
                save_pred(epoch_ctx.stored[store_key], checkpoint_folder=globalvars.main_context.checkpoint_dir, pred_file=pred_file)

    def _set_offset_learning_rate(self, epoch, step):
        if self.offset_optimizer is None:
            return

        if step >= hparams.TRAIN.OFFSET.TRAIN_MIN_STEP and hparams.TRAIN.OFFSET.DECAY_STEP_LR > 0 and hparams.TRAIN.OFFSET.GAMMA_LR > 0:
            step_offset = max(0, step - hparams.TRAIN.OFFSET.TRAIN_MIN_STEP)
        else:
            step_offset = -1

        for param_group in self.offset_optimizer.param_groups:
            if step_offset >= 0:
                cur_lr_offset = param_group["init_lr"] * (hparams.TRAIN.OFFSET.GAMMA_LR ** (float(step_offset) / hparams.TRAIN.OFFSET.DECAY_STEP_LR))
                log_i("Set {} to {:.5f}".format(param_group["para_name"], cur_lr_offset))
            else:
                cur_lr_offset = param_group["init_lr"]
            param_group["lr"] = cur_lr_offset

    def _update_training_state(self, iteration, step, only_on_boundary):
        if step >= hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            if hparams.TRAIN.OFFSET.SEP_TRAIN_ITER > 0:
                if iteration >= hparams.TRAIN.OFFSET.SEP_TRAIN_ITER:
                    if not only_on_boundary or iteration == hparams.TRAIN.OFFSET.SEP_TRAIN_ITER:
                        self._set_training_state(update_weight=False, update_offset=True, update_transformer=True)
                else:
                    if not only_on_boundary or iteration == 0:
                        self._set_training_state(update_weight=True, update_offset=False, update_transformer=False)
            elif not only_on_boundary or step == hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
                self._set_training_state(update_offset=True, update_transformer=True)
        else:
            if not only_on_boundary or step == 0:
                self._set_training_state(update_offset=False, update_transformer=False)

    def epoch_start(self, epoch, step, evaluate_only):
        if not evaluate_only:
            cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams.TRAIN.LEARNING_RATE, hparams.TRAIN.SCHEDULE, hparams.TRAIN.GAMMA_LR) 
            log_i("Set learning rate to {:.5f}".format(cur_lr))
            if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
                self._set_offset_learning_rate(epoch, step)

            self._update_training_state(0, step, False)

            if hparams.TRAIN.GRADUAL_SIZE:
                # Set gradual data size
                if isinstance(self.train_dataset, Subset):
                    train_dataset = self.train_dataset.dataset
                else:
                    train_dataset = self.train_dataset
                if epoch <= 40:
                    train_dataset.set_resize_scale(0.25 + 0.75 * (epoch - 1) / 40)
                else:
                    train_dataset.set_resize_scale(1)
            if epoch == 1:
                tb_writer = globalvars.main_context.get("tb_writer")
                if tb_writer:
                    custom_scalars = {
                        "loss": {
                            "all": ["Multiline", [r"loss/all_.*"]],
                            "train": ["Multiline", [r"loss/all_train", r"loss/(?!all_).*"]]
                        },
                        "offset": {
                            "move_dis": ["Multiline", [r"move_dis/(?!right_ratio)"]],
                            "sigma_change": ["Multiline", [r"sigma_change/(?!right_ratio)"]]
                        }
                    }
                    if self.data_source == "coco":
                        custom_scalars.update({
                            "coco": {
                                "AP": ["Multiline", [r"coco/AP_.*"]],
                                "AR": ["Multiline", [r"coco/AR_.*"]]
                            }
                        })
                    elif self.data_source == "mpii":
                        custom_scalars.update({
                            "mpii": {
                                "PCKh": ["Multiline", [r"mpii/.*"]]
                            }
                        })
                    else:
                        raise ValueError("Unknown data_source")
                    tb_writer.add_custom_scalars(custom_scalars)

    def _save_offsets(self, step):
        offset_disabled = True
        for dm in globalvars.displace_mods:
            if dm.learnable_offset:
                offset_disabled = False
        if not offset_disabled:
            torch.save([(dm.get_all_offsets(detach=True) * dm.offset_scale).cpu() for dm in globalvars.displace_mods], os.path.join(globalvars.main_context.checkpoint_dir, "offset_{}.pth".format(step)))
            if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
                torch.save([dp.sigma.cpu() for dp in globalvars.dpools], os.path.join(globalvars.main_context.checkpoint_dir, "dpool_{}.pth".format(step)))

    def epoch_end(self, epoch, step, evaluate_only):
        if not evaluate_only and self.offset_optimizer:
            self._save_offsets(step)

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        optimize_weight = self.update_weight
        optimize_offset = bool(self.offset_optimizer and self.update_offset)
        optimize_transformer = bool(self.transformer_optimizer and self.update_transformer)

        if optimize_weight:
            self.optimizer.zero_grad()
        if optimize_offset:
            self.offset_optimizer.zero_grad()
        if optimize_transformer:
            self.transformer_optimizer.zero_grad()
        loss.backward()
        if optimize_weight:
            self.optimizer.step()
        if optimize_offset:
            self.offset_optimizer.step()
            if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
                for idp, dp in enumerate(globalvars.dpools):
                    sigma_data = dp.sigma.data
                    sigma_data[sigma_data.abs() > dp.max_sigma] = dp.max_sigma
        if optimize_transformer:
            self.transformer_optimizer.step()
            for dm in globalvars.displace_mods:
                if dm.offset_transformer is not None and dm.offset_transformer.effect_scale is not None:
                    dm.offset_transformer.effect_scale.data.add_(dm.offset_transformer.scale_grow_step.data).clamp_(0, 1)

    def _set_training_state(self, update_weight=None, update_offset=None, update_transformer=None):
        def set_requires_grad(paras, requires_grad):
            for para in paras:
                para.requires_grad = requires_grad

        state_ori = dict()
        log_i("Set training state: update_weight={}, update_offset={}, update_transformer={}".format(update_weight, update_offset, update_transformer))

        if update_weight is not None:
            state_ori["update_weight"] = self.update_weight
            self.update_weight = update_weight
            set_requires_grad(self.general_parameters, update_weight)

        if update_offset is not None:
            state_ori["update_offset"] = self.update_offset
            self.update_offset = update_offset
            set_requires_grad(self.offset_parameters, update_offset)
            set_requires_grad(self.offset_regressor_parameters, update_offset)
            set_requires_grad(self.dpool_parameters, update_offset)

        if update_transformer is not None:
            state_ori["update_transformer"] = self.update_transformer
            self.update_transformer = update_transformer
            set_requires_grad(self.offset_transformer_parameters, update_transformer)

        return state_ori

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, progress: dict, train: bool) -> torch.Tensor:
        image_ids = batch["img_index"].tolist()
        img = batch["img"]
        if not hparams.MODEL.REGRESS_PREDICT:
            det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        img_ori_size = batch["img_ori_size"]
        keypoint = batch["keypoint"]
        batch_size = img.size(0)
        globalvars.progress = progress

        if train:
            self._update_training_state(progress["iter"], progress["step"], True)
            if self.offset_optimizer and progress["step"] == hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
                self._save_offsets(progress["step"])

        img_cuda = img.cuda(non_blocking=True)
        if not hparams.MODEL.REGRESS_PREDICT:
            det_map_gt_cuda = det_maps_gt.cuda(non_blocking=True)
        keypoint_cuda = keypoint.cuda(non_blocking=True)
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img_cuda

        output_maps = self.model(img_cuda)

        mask_notlabeled = (keypoint_cuda[:, :, 2] <= 0.1)
        mask_labeled = (~mask_notlabeled)
        mask_visible = (keypoint_cuda[:, :, 2] > 1.1)
        mask_notvisible = (mask_labeled & (~mask_visible))

        if hparams.MODEL.DETAIL.LOSS_FINAL == "all":
            masking_final = 1.
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "labeled":
            masking_final = mask_labeled.float().view(-1, self.num_parts, 1, 1)
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "visible":
            masking_final = mask_visible.float().view(-1, self.num_parts, 1, 1)
        else:
            assert False

        if not hparams.MODEL.REGRESS_PREDICT:
            loss_map = ((output_maps - det_map_gt_cuda).pow(2) * \
                masking_final).mean().sqrt()
        else:
            kp_pred_reg = self.pose_regressor(output_maps)
            loss_map = ((kp_pred_reg - keypoint_cuda[:, :, :2]).norm(dim=-1)[:, :, None, None] * masking_final).mean()

        epoch_ctx.set_iter_data("loss_map", loss_map)

        loss_dpool = 0
        if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1 and hparams.MODEL.LOSS_DPOOL_COF is not None:
            for idp, dp in enumerate(globalvars.dpools):
                loss_dpool = loss_dpool + dp.sigma.pow(2).mean()
            loss_dpool = loss_dpool * hparams.MODEL.LOSS_DPOOL_COF
            epoch_ctx.set_iter_data("loss_dpool", loss_dpool)

        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        loss = loss_map + loss_dpool

        epoch_ctx.set_iter_data("loss", loss)

        if not train or config.vis:
            kp_pred, score = parse_map(output_maps, thres=hparams.EVAL.PARSE_THRESHOLD)
            kp_pred_affined = kp_pred.copy()
            for samp_i in range(batch_size):
                kp_pred_affined[samp_i, :, :2] = kpt_affine(kp_pred_affined[samp_i, :, :2] * FACTOR, np.linalg.pinv(transform_mat[samp_i])[:2])
                if img_flipped[samp_i]:
                    kp_pred_affined[samp_i] = fliplr_pts(kp_pred_affined[samp_i], self.flip_index, width=img_ori_size[samp_i, 0].item())
            if self.data_source == "coco":
                ans = generate_ans(image_ids, kp_pred_affined, score)
                epoch_ctx.add_store("annotates", {"image_index": image_ids, "annotate": ans})
            else:
                ans = generate_mpii_ans(image_ids, batch["person_index"], kp_pred_affined)
                epoch_ctx.add_store("annotates", {"image_index": image_ids, "annotate": ans, "pred": torch.from_numpy(kp_pred_affined), "gt": batch["keypoint_ori"], "head_box": batch["head_box"]})

            if config.store and hparams.CONFIG.STORE_MAP and train:
                if not hasattr(epoch_ctx, "store_counter"):
                    epoch_ctx.store_counter = 0
                if epoch_ctx.store_counter < 30:
                    epoch_ctx.add_store("pred", {"image_index": image_ids, "img": np.ascontiguousarray(self.train_dataset.restore_image(img.data.cpu().numpy())), "gt": det_maps_gt, "pred": output_maps})
                epoch_ctx.store_counter += 1

        if config.vis and False:
            import matplotlib.pyplot as plt
            img_restored = np.ascontiguousarray(self.train_dataset.restore_image(img.data.numpy())[..., ::-1])

            if False:
                nrows = int(np.sqrt(float(batch_size)))
                ncols = (batch_size + nrows - 1) // nrows
                fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                for ax in axes.flat:
                    ax.axis("off")
                for i in range(batch_size):
                    draw_img = img_restored[i].copy()
                    for j in range(self.num_parts):
                        pt = kp_pred_affined[i, j]
                        if pt[2] > 0:
                            cv2.circle(draw_img, (int(pt[0] * FACTOR), int(pt[1] * FACTOR)), radius=2, color=(0, 0, 255), thickness=-1)
                    axes.flat[i].imshow(draw_img[..., ::-1])

            if False:
                for i in range(min(1, batch_size)):
                    nrows = 3; ncols = 6

                    pred_resized = batch_resize((output_maps[i].data.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8) , img.size()[-2:])
                    
                    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                    for ax in axes.flat:
                        ax.axis("off")
                    for j in range(self.num_parts):
                        ax = axes.flat[j]
                        draw_img = cv2.addWeighted(img_restored[i], 1, cv2.applyColorMap(pred_resized[j, :, :, None], cv2.COLORMAP_HOT), 0.5, 0)
                        ax.imshow(draw_img[..., ::-1])
                        ax.set_title(datasets.mscoco.PART_LABELS[j])
                    plt.show()

        return loss

    def summarize_iter(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")

        if "loss_dpool" in epoch_ctx.iter_data:
            loss_dpool_val = epoch_ctx.iter_data["loss_dpool"].item()
        else:
            loss_dpool_val = None

        loss_map_val = epoch_ctx.iter_data["loss_map"].item()
        loss_val = epoch_ctx.iter_data["loss"].item()
        if train and tb_writer is not None:
            tb_writer.add_scalar("loss/map", loss_map_val, progress["step"])
            if loss_dpool_val is not None:
                tb_writer.add_scalar("loss/dpool", loss_dpool_val, progress["step"])
            tb_writer.add_scalar("loss/all_train", loss_val, progress["step"])

        epoch_ctx.add_scalar("loss", loss_val)

        if train and self.offset_optimizer and self.update_offset:
            if tb_writer:
                move_dis_avg = list()
                move_dis = list()
                for idm in range(len(globalvars.displace_mods)):
                    dm = globalvars.displace_mods[idm]
                    if dm.offset.size(0) == 0:
                        continue
                    self.move_dis_avgmeter[idm].update((dm.offset.detach() * dm.offset_scale).cpu())
                    move_dis_avg.append(self.move_dis_avgmeter[idm].avg)
                    move_dis.append(self.move_dis_avgmeter[idm].lastdiff)
                move_dis_avg = np.mean(move_dis_avg)
                move_dis = np.mean(move_dis)

                tb_writer.add_scalar("{}/{}".format("move_dis", "cur"), move_dis, progress["step"])
                tb_writer.add_scalar("{}/{}".format("move_dis", "avg"), move_dis_avg, progress["step"])
                tb_writer.add_scalar("{}/{}".format("move_dis", "right_ratio"), move_dis_avg / move_dis, progress["step"])

                if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
                    sigma_change = list()
                    sigma_change_avg = list()
                    for idp, dp in enumerate(globalvars.dpools):
                        self.change_sigma_avgmeter[idp].update(dp.sigma.detach().cpu().abs())
                        sigma_change.append(self.change_sigma_avgmeter[idp].lastdiff_dir)
                        sigma_change_avg.append(self.change_sigma_avgmeter[idp].avg_dir)
                    sigma_change = np.mean(sigma_change)
                    sigma_change_avg = np.mean(sigma_change_avg)
                    tb_writer.add_scalar("{}/{}".format("sigma_change", "cur"), sigma_change, progress["step"])
                    tb_writer.add_scalar("{}/{}".format("sigma_change", "avg"), sigma_change_avg, progress["step"])
                    tb_writer.add_scalar("{}/{}".format("sigma_change", "right_ratio"), sigma_change_avg / sigma_change, progress["step"])

            if (progress["step"] + 1) % hparams.LOG.OFFSET_SAVE_INTERVAL == 0:
                self._save_offsets(progress["step"] + 1)

    def summarize_epoch(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")
        if tb_writer and not train:
            tb_writer.add_scalar("loss/all_valid", epoch_ctx.scalar["loss"].avg, progress["step"])

class Attention(nn.Module):
    def __init__(self, inplanes, outplanes, input_shape=None, bias_planes=0, bias_factor=0, space_norm=True, stride=1):
        super(Attention, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.total_inplanes = inplanes
        self.input_shape = input_shape
        self.bias_planes = bias_planes
        self.bias_factor = bias_factor
        if input_shape is not None and bias_planes > 0 and bias_factor > 0:
            self.total_inplanes += bias_planes
            bias_shape = (int(input_shape[0] // bias_factor), int(input_shape[1] // bias_factor))
            if config.check:
                log_i("bias_shape = {}".format(str(bias_shape)))
            self.bias = nn.Parameter(torch.ones(1, bias_planes, bias_shape[0], bias_shape[1], dtype=torch.float))
        else:
            self.bias = None

        self.space_norm = space_norm
        if space_norm:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN),
                nn.Softplus(),
                SpaceNormalization())
        else:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN),
                nn.Sigmoid())

    def forward(self, x):
        if self.bias is not None:
            atten_bias = nn.functional.interpolate(self.bias, size=x.size()[-2:], mode="bilinear", align_corners=True).expand(x.size(0), -1, -1, -1)
            if self.inplanes > 0:
                x = torch.cat([x, atten_bias], dim=1)
            else:
                x = atten_bias

        return self.atten(x)

class DynamicPooling(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(DynamicPooling, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1
        x = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).view(1, -1).expand(kernel_size, -1)
        y = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).view(-1, 1).expand(-1, kernel_size)
        dissq = torch.stack((x, y), dim=0).pow(2).float().sum(dim=0)
        self.register_buffer("dissq", dissq)
        self.sigma = nn.Parameter(torch.zeros(num_channels))
        self.sigma.data.fill_(kernel_size / 2 / 2)
        self.register_buffer("max_sigma", torch.tensor(kernel_size / 2, dtype=torch.float))
        self.eps = np.finfo(np.float32).eps.item()
        globalvars.dpools.append(self)

    def forward(self, x):
        kernel = torch.exp(-(self.dissq / 2)[None] / (self.sigma.pow(2)[:, None, None] + self.eps))
        kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        kernel = kernel.view(self.num_channels, 1, self.kernel_size, self.kernel_size)
        expx = torch.exp(x.clamp(max=88.722835))
        gp_expx = F.conv2d(expx, kernel, padding=(self.kernel_size // 2, self.kernel_size // 2), groups=self.num_channels)
        pooled = torch.log(gp_expx + self.eps)
        return pooled

class SequentialForOffsetBlockTransformer(nn.Sequential):
    def forward(self, input, extra):
        for module in self._modules.values():
            if isinstance(module, OffsetBlock):
                if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
                    assert extra is not None
                    input = module(input, transformer_source=extra, atten_source=extra)
                else:
                    assert extra is None
                    input = module(input, transformer_source=input, atten_source=input)
            else:
                input = module(input)
        return input

class OffsetBlock(nn.Module):
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1, use_transformer=False, use_atten=False, use_post_atten=False, independent_atten_source=False):
        super(OffsetBlock, self).__init__()
        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        if hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE > 0:
            init_num_x = 2 * self.out_width // hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE
            init_num_y = 2 * self.out_height // hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE
            assert init_num_x > 0 and init_num_y > 0, "Insufficient number of init offsets"
            init_num_off = init_num_x * init_num_y
            channels_per_off = int(np.round(displace_planes / init_num_off).item())
            displace_planes_new = channels_per_off * init_num_off
            log_i("Displace plane number rounded from {} to {}".format(displace_planes, displace_planes_new))
            displace_planes = displace_planes_new
        self.displace_planes = displace_planes
        if use_transformer:
            offset_transformer = OffsetTransformer(
                hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT else self.inplanes,
                self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN,
                bottleneck=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.BOTTLENECK,
                single_regress=False,
                scale_grow_step=1 / hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None,
                absolute_regressor=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ABSOLUTE_REGRESSOR)
        else:
            offset_transformer = None
        self.displace = DisplaceChannel(
            self.out_height, self.out_width,
            self.displace_planes, self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=hparams.MODEL.DETAIL.LEARNABLE_OFFSET,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transformer=offset_transformer,
            half_reversed_offset=hparams.MODEL.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=globalvars.displace_mods[-1] if hparams.MODEL.LEARNABLE_OFFSET.REUSE_OFFSET and len(globalvars.displace_mods) > 0 else None)
        if hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE > 0:
            width_span = hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE * (init_num_x - 1)
            height_span = hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE * (init_num_y - 1)
            for iy, off_y in enumerate(torch.linspace(-height_span / 2, height_span / 2, steps=init_num_y)):
                for ix, off_x in enumerate(torch.linspace(-width_span / 2, width_span / 2, steps=init_num_x)):
                    ichan = (iy * init_num_x + ix) * channels_per_off
                    self.displace.offset.data[ichan:ichan+channels_per_off, 0] = off_x / self.displace.offset_scale
                    self.displace.offset.data[ichan:ichan+channels_per_off, 1] = off_y / self.displace.offset_scale

        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if use_atten:
            self.atten_displace = Attention(
                self.inplanes if not independent_atten_source else hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE,
                self.displace_planes,
                input_shape=(self.height, self.width),
                bias_planes=0,
                bias_factor=0,
                space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN_SPACE_NORM,
                stride=stride)
        else:
            self.atten_displace = None
        if use_post_atten:
            self.atten_post = Attention(self.inplanes, self.inplanes, input_shape=(self.out_height, self.out_width), bias_planes=0, bias_factor=0, space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN_SPACE_NORM, stride=stride)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None
        
        if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
            self.dpool = DynamicPooling(self.displace_planes, hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE)
        else:
            self.dpool = None

    def forward(self, x, transformer_source=None, atten_source=None):
        if not hparams.TRAIN.OFFSET.ALWAYS_TRAIN_BLOCK and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        shortcut = x
        out_pre = self.pre_offset(x)

        if self.dpool:
            out_pre = self.dpool(out_pre)

        out_dis = self.displace(out_pre, transformer_source=transformer_source)

        if self.atten_displace is not None:
            out_atten = self.atten_displace(atten_source)
            out_dis = out_dis * out_atten
        else:
            out_atten = None

        out_post = self.post_offset(out_dis)

        if self.atten_post is not None:
            out_post = out_post * self.atten_post(x)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_skip = shortcut + out_post

        out_final = self.relu(self.bn(out_skip))

        return out_final

class MyPose(nn.Module):
    def __init__(self, num_class):
        super(MyPose, self).__init__()
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE and hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
            self.transformer = TransformFeature()
        else:
            self.transformer = None
        self.estimator = SimpleEstimator(num_class)

    def forward(self, x):
        if self.transformer is not None:
            transform_features = self.transformer(x)
        else:
            transform_features = None
        prediction = self.estimator(x, transform_features)
        return prediction

class TransformFeature(nn.Module):
    def __init__(self):
        super(TransformFeature, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.offblk = OffsetBlock(
            hparams.MODEL.INP_SHAPE[1] // 4,
            hparams.MODEL.INP_SHAPE[0] // 4,
            64,
            hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE,
            hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_OFFSET,
            use_transformer=False,
            use_atten=False)

    def forward(self, x):
        return self.offblk(self.pre(x))

class RegressPredictor(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        x = torch.arange(width, dtype=torch.float) + 0.5
        y = torch.arange(height, dtype=torch.float) + 0.5
        self.register_buffer("weight_x", x.view(1, -1).repeat(height, 1))
        self.register_buffer("weight_y", y.view(-1, 1).repeat(1, width))
    
    def forward(self, inp):
        assert inp.size()[-2:] == (self.height, self.width)
        inp_exp_norm = inp.exp()
        inp_exp_norm = inp_exp_norm / inp_exp_norm.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        x = (inp_exp_norm * self.weight_x.expand_as(inp_exp_norm)).sum(dim=-1).sum(dim=-1)
        y = (inp_exp_norm * self.weight_y.expand_as(inp_exp_norm)).sum(dim=-1).sum(dim=-1)
        return torch.stack([x, y], dim=-1)

class SimpleEstimator(nn.Module):
    CHANNELS = 256
    def __init__(self, num_class):
        super(SimpleEstimator, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        channels = 64
        offblks = []
        for i in range(hparams.MODEL.LEARNABLE_OFFSET.NUM_BLK):
            offblks.append(
                OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // 4,
                    hparams.MODEL.INP_SHAPE[0] // 4,
                    channels,
                    SimpleEstimator.CHANNELS,
                    hparams.MODEL.LEARNABLE_OFFSET.NUM_OFFSET,
                    use_atten=hparams.MODEL.LEARNABLE_OFFSET.ENABLE_ATTEN,
                    independent_atten_source=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE,
                    use_transformer=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE))
            channels = SimpleEstimator.CHANNELS
        self.offblk = SequentialForOffsetBlockTransformer(*offblks)
        
        self.predictor = self._make_predictor(SimpleEstimator.CHANNELS, num_class)

    def _make_predictor(self, planes, num_class):
        layers = []
        layers.append(nn.Conv2d(planes, planes,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(planes))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(planes, num_class,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x, transform_features):
        if not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE or not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
            assert transform_features is None
        else:
            assert transform_features is not None

        off = self.offblk(self.pre(x), transform_features)

        return self.predictor(off)
