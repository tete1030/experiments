import os
import re
import math
import time
import itertools
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from torch.utils.data import Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import lib.datasets as datasets
from lib.utils.transforms import fliplr_pts, kpt_affine
from lib.utils.evaluation import accuracy, OffsetCycleAverageMeter, parse_map, generate_ans, generate_mpii_ans
from lib.utils.imutils import batch_resize
from utils.globals import config, hparams, globalvars
from utils.log import log_i
from utils.train import adjust_learning_rate
from utils.checkpoint import save_pred, load_pretrained_loose, save_checkpoint, RejectLoadError
from utils.miscs import nprand_init
from experiments.baseexperiment import BaseExperiment, EpochContext
from .resnet import resnet18, resnet50, resnet101
from .offset import OffsetBlock

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
        else:
            self.move_dis_avgmeter = None

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
        globalvars.early_predictor_size = list()
        globalvars.displace_mods = list()
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            globalvars.early_predictors = list()
            globalvars.pre_early_predictor_outs = dict()

        if hparams.MODEL.USE_GN:
            globalvars.BatchNorm2dImpl = GroupNormWrapper
        else:
            globalvars.BatchNorm2dImpl = nn.BatchNorm2d

        assert hparams.MODEL.DETAIL.EARLY_PREDICTOR or not hparams.MODEL.DETAIL.FIRST_ESP_ONLY

        pretrained = hparams.MODEL.RESNET_PRETRAINED
        if config.resume:
            pretrained = None

        self.model = nn.DataParallel(Controller(MainModel(hparams.MODEL.OUT_SHAPE[::-1], self.num_parts, pretrained=pretrained), self.num_parts).cuda())
        assert OffsetBlock._counter == len(hparams.MODEL.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO) or not hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK

    def init_optimizer(self):
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in globalvars.displace_mods if hasattr(dm, "offset")]))
            self.offset_regressor_parameters = list(filter(lambda x: x.requires_grad, list(itertools.chain.from_iterable([dm.offset_regressor.parameters() for dm in globalvars.displace_mods if hasattr(dm, "offset_regressor")]))))
        else:
            self.offset_parameters = []
            self.offset_regressor_parameters = []

        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            self.early_predictor_parameters = list(filter(lambda x: x.requires_grad, itertools.chain.from_iterable([ep.parameters() for ep in globalvars.early_predictors])))
        else:
            self.early_predictor_parameters = []

        special_parameter_ids = list(map(lambda x: id(x), self.offset_parameters + self.offset_regressor_parameters + self.early_predictor_parameters))
        self.normal_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))

        self.optimizer = torch.optim.Adam(
            self.normal_parameters,
            lr=hparams.TRAIN.LEARNING_RATE,
            weight_decay=hparams.TRAIN.WEIGHT_DECAY)

        offset_optimizer_args = []
        if len(self.offset_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_lr", "params": self.offset_parameters, "lr": hparams.TRAIN.OFFSET.LR, "init_lr": hparams.TRAIN.OFFSET.LR})
        if len(self.offset_regressor_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_regressor_lr", "params": self.offset_regressor_parameters, "lr": hparams.TRAIN.OFFSET.LR_REGRESSOR, "init_lr": hparams.TRAIN.OFFSET.LR_REGRESSOR})
        if len(offset_optimizer_args) > 0:
            self.offset_optimizer = torch.optim.Adam(offset_optimizer_args)
        else:
            self.offset_optimizer = None

        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            self.early_predictor_optimizer = torch.optim.Adam(
                self.early_predictor_parameters,
                lr=hparams.TRAIN.LEARNING_RATE,
                weight_decay=hparams.TRAIN.WEIGHT_DECAY)
        else:
            self.early_predictor_optimizer = None

    def init_dataloader(self):
        self.worker_init_fn = nprand_init
        super().init_dataloader()

    def init_mscoco(self):
        self.coco = COCO("data/mscoco/person_keypoints_train2017.json")
        self.train_dataset = datasets.COCOSinglePose(
            "data/mscoco/images2017",
            self.coco,
            "data/mscoco/sp_split_2017.pth",
            "data/mscoco/" + hparams.DATASET["COCO"].MEAN_STD_FILE,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            ext_border=hparams.DATASET["COCO"].EXT_BORDER,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
            scale_factor=hparams.DATASET["COCO"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["COCO"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["COCO"].TRANSLATION_FACTOR)
        self.train_collate_fn = datasets.COCOSinglePose.collate_function
        self.valid_collate_fn = datasets.COCOSinglePose.collate_function

    def init_mpii(self):
        self.train_dataset = datasets.MPII("data/mpii/images",
            "data/mpii/mpii_human_pose.json",
            "data/mpii/split_sig.pth",
            "data/mpii/mean_std.pth",
            True,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            if self.early_predictor_optimizer:
                self.early_predictor_optimizer.load_state_dict(checkpoint["early_predictor_optimizer"])
        if self.offset_optimizer:
            self.move_dis_avgmeter = checkpoint["move_dis_avgmeter"]
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_full, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict() if self.offset_optimizer else None,
            "early_predictor_optimizer": self.early_predictor_optimizer.state_dict() if self.early_predictor_optimizer else None,
            "move_dis_avgmeter": self.move_dis_avgmeter if not hparams.MODEL.DETAIL.DISABLE_DISPLACE else None
        }
        save_checkpoint(checkpoint_dict, checkpoint_full=checkpoint_full, force_replace=True)

    def evaluate(self, epoch_ctx:EpochContext, epoch, step):
        if "annotates" not in epoch_ctx.stored:
            return

        try:
            tb_writer = globalvars.main_context.tb_writer
        except KeyError:
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
                    tb_writer.add_scalar("{}/{}/{}".format(hparams.LOG.TB_DOMAIN, stat_type, stat_name), stats[istat], step)

        elif self.data_source == "mpii":
            annotates = epoch_ctx.stored["annotates"]
            acc = accuracy(annotates["pred"], annotates["gt"], annotates["head_box"])
            if tb_writer:
                tb_writer.add_scalar("{}/{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh", "avg"), float(acc[0]), step)
            results = list()
            results.append("avg: {:2.2f}".format(float(acc[0]) * 100))
            for i in range(0, acc.size(0)-1):
                if tb_writer:
                    tb_writer.add_scalar("{}/{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh", datasets.mpii.PART_LABELS[i]), float(acc[i+1]), step)
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

    def set_offset_learning_rate(self, epoch, step):
        if self.offset_optimizer is None:
            return

        if step >= hparams.TRAIN.OFFSET.TRAIN_MIN_STEP and hparams.TRAIN.OFFSET.LR_DECAY_STEP > 0 and hparams.TRAIN.OFFSET.LR_GAMMA > 0:
            step_offset = max(0, step - hparams.TRAIN.OFFSET.TRAIN_MIN_STEP)
        else:
            step_offset = -1

        for param_group in self.offset_optimizer.param_groups:
            if step_offset >= 0:
                cur_lr_offset = param_group["init_lr"] * (hparams.TRAIN.OFFSET.LR_GAMMA ** (float(step_offset) / hparams.TRAIN.OFFSET.LR_DECAY_STEP))
                log_i("Set {} to {:.5f}".format(param_group["para_name"], cur_lr_offset))
            else:
                cur_lr_offset = param_group["init_lr"]
            param_group["lr"] = cur_lr_offset

    def epoch_start(self, epoch, step, evaluate_only):
        if not evaluate_only:
            cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams.TRAIN.LEARNING_RATE, hparams.TRAIN.SCHEDULE, hparams.TRAIN.LR_GAMMA)
            log_i("Set learning rate to {:.5f}".format(cur_lr))
            adjust_learning_rate(self.early_predictor_optimizer, epoch, hparams.TRAIN.LEARNING_RATE, hparams.TRAIN.SCHEDULE, hparams.TRAIN.LR_GAMMA)
            if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
                self.set_offset_learning_rate(epoch, step)

    def save_offsets(self, step):
        offset_disabled = True
        for dm in globalvars.displace_mods:
            if dm.learnable_offset:
                offset_disabled = False
        if not offset_disabled:
            torch.save([(dm.get_all_offsets(detach=True) * dm.offset_scale).cpu() for dm in globalvars.displace_mods], os.path.join(globalvars.main_context.checkpoint_dir, "offset_{}.pth".format(step)))

    def epoch_end(self, epoch, step, evaluate_only):
        if not evaluate_only and not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.save_offsets(step)

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        optimize_offset = False
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer is not None and progress["step"] >= hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            optimize_offset = True

        self.optimizer.zero_grad()
        if optimize_offset:
            self.offset_optimizer.zero_grad()
        if self.early_predictor_optimizer:
            self.early_predictor_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if optimize_offset:
            self.offset_optimizer.step()
        if self.early_predictor_optimizer:
            self.early_predictor_optimizer.step()

        if optimize_offset:
            move_dis_avg = list()
            move_dis = list()
            for idm in range(len(globalvars.displace_mods)):
                dm = globalvars.displace_mods[idm]
                if dm.offset.size(0) == 0:
                    continue
                self.move_dis_avgmeter[idm].update((dm.offset.detach() * dm.offset_scale).cpu())
                move_dis_avg.append(self.move_dis_avgmeter[idm].avg)
                move_dis.append(self.move_dis_avgmeter[idm].lastdiff)
            globalvars.main_context.tb_writer.add_scalar("{}/{}/{}".format(hparams.LOG.TB_DOMAIN, "move_dis", "mod"), np.mean(move_dis_avg), progress["step"] + 1)
            globalvars.main_context.tb_writer.add_scalar("{}/{}/{}".format(hparams.LOG.TB_DOMAIN, "move_dis", "mod_cur"), np.mean(move_dis), progress["step"] + 1)

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer is not None and (progress["step"] + 1) % hparams.LOG.OFFSET_SAVE_INTERVAL == 0:
            self.save_offsets(progress["step"] + 1)

    def summarize_gradient(self):
        # TODO: all gradients? offset gradients?
        pass

    def summarize_parameter(self):
        # TODO: summarize all parameters
        pass

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, progress: dict, train: bool) -> torch.Tensor:
        image_ids = batch["img_index"].tolist()
        img = batch["img"]
        det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        img_ori_size = batch["img_ori_size"]
        keypoint = batch["keypoint"]
        batch_size = img.size(0)
        globalvars.progress = progress

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer and progress["step"] == hparams.TRAIN.OFFSET.TRAIN_MIN_STEP and progress["train"]:
            self.save_offsets(progress["step"])

        det_map_gt_cuda = [dm.cuda() for dm in det_maps_gt]
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img
        output_maps, early_predictor_outputs = self.model(img)
        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        mask_notlabeled = (keypoint[:, :, 2] <= 0.1).cuda()
        mask_labeled = (~mask_notlabeled)
        mask_visible = (keypoint[:, :, 2] > 1.1).cuda()
        mask_notvisible = (mask_labeled & (~mask_visible))

        if hparams.MODEL.DETAIL.LOSS_EARLY == "all":
            masking_early = 1.
        elif hparams.MODEL.DETAIL.LOSS_EARLY == "labeled":
            masking_early = mask_labeled.float().view(-1, self.num_parts, 1, 1)
        elif hparams.MODEL.DETAIL.LOSS_EARLY == "visible":
            masking_early = mask_visible.float().view(-1, self.num_parts, 1, 1)
        else:
            assert False

        if hparams.MODEL.DETAIL.LOSS_FINAL == "all":
            masking_final = 1.
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "labeled":
            masking_final = mask_labeled.float().view(-1, self.num_parts, 1, 1)
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "visible":
            masking_final = mask_visible.float().view(-1, self.num_parts, 1, 1)
        else:
            assert False

        if not hparams.MODEL.DETAIL.FIRST_ESP_ONLY:
            loss = 0.
            for ilabel, (outv, gtv) in enumerate(zip(output_maps, det_map_gt_cuda)):
                # if ilabel < len(det_map_gt_cuda) - 1:
                #     gtv *= (keypoint[:, :, 2] > 1.1).float().view(-1, self.num_parts, 1, 1).cuda()
                if ilabel < len(det_map_gt_cuda) - 1:
                    loss = loss + ((outv - gtv).pow(2) * masking_early).mean().sqrt()
                else:
                    loss = loss + ((outv - gtv).pow(2) * masking_final).mean().sqrt()

            if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
                assert len(early_predictor_outputs) == len(hparams.MODEL.DETAIL.EARLY_PREDICTOR_LABEL_INDEX)
                for ilabel, outv in enumerate(early_predictor_outputs):
                    loss = loss + ((outv - det_map_gt_cuda[hparams.MODEL.DETAIL.EARLY_PREDICTOR_LABEL_INDEX[ilabel]]).pow(2) * \
                        masking_early).mean().sqrt()
        else:
            loss = ((early_predictor_outputs[0] - det_map_gt_cuda[hparams["model"]["detail"]["early_predictor_label_index"][0]]).pow(2) * \
                masking_early).mean().sqrt()

        epoch_ctx.add_scalar("loss", loss.item())

        if (loss.data != loss.data).any():
            import ipdb; ipdb.set_trace()

        if not train or config.vis:
            kp_pred, score = parse_map(output_maps[-1], thres=hparams.EVAL.PARSE_THRESHOLD)
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
            img_restored = np.ascontiguousarray(self.train_dataset.restore_image(img.data.cpu().numpy())[..., ::-1])

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
                    for i_out in range(len(output_maps)):
                        pred_resized = batch_resize((output_maps[i_out][i].data.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8) , img.size()[-2:])
                        
                        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                        fig.suptitle("%d" % (i_out,))
                        for ax in axes.flat:
                            ax.axis("off")
                        for j in range(self.num_parts):
                            ax = axes.flat[j]
                            draw_img = cv2.addWeighted(img_restored[i], 1, cv2.applyColorMap(pred_resized[j, :, :, None], cv2.COLORMAP_HOT), 0.5, 0)
                            ax.imshow(draw_img[..., ::-1])
                            ax.set_title(datasets.mscoco.PART_LABELS[j])
                    plt.show()

        return loss

# Attention List:
# - middle_outputs
# - BreakableSequential in Blocks
# - 

class Controller(nn.Module):
    def __init__(self, main_model, num_parts):
        super(Controller, self).__init__()
        self.main_model = main_model
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            early_predictor = list()
            assert len(globalvars.early_predictor_size) > 0
            for inplanes, inshape_factor in globalvars.early_predictor_size:
                early_predictor.append(
                    Predictor(
                        inplanes, 
                        (hparams.MODEL.INP_SHAPE[1] // inshape_factor, hparams.MODEL.INP_SHAPE[0] // inshape_factor),
                        hparams.MODEL.OUT_SHAPE[::-1],
                        num_parts))
            self.early_predictor = nn.ModuleList(early_predictor)
        else:
            self.early_predictor = None

    def forward(self, x):
        if self.early_predictor:
            globalvars.pre_early_predictor_outs[x.device] = list()

        out = self.main_model(x)

        if self.early_predictor:
            pre_early_predictor_outs = globalvars.pre_early_predictor_outs[x.device]
            globalvars.pre_early_predictor_outs[x.device] = list()
            assert hparams.MODEL.DETAIL.FIRST_ESP_ONLY or len(pre_early_predictor_outs) == len(self.early_predictor)
            return out, [self.early_predictor[i](pre_early_predictor_outs[i]) for i in range(len(pre_early_predictor_outs))]
        else:
            return out, None

class MainModel(nn.Module):
    def __init__(self, output_shape, num_points, pretrained=None):
        super(MainModel, self).__init__()
        if hparams.MODEL.RESNET == 18:
            self.resnet = resnet18(pretrained=pretrained)
            channel_settings = [512, 256, 128, 64]
        elif hparams.MODEL.RESNET == 50:
            self.resnet = resnet50(pretrained=pretrained)
            channel_settings = [2048, 1024, 512, 256]
        elif hparams.MODEL.RESNET == 101:
            self.resnet = resnet101(pretrained=pretrained)
            channel_settings = [2048, 1024, 512, 256]
        else:
            assert False
        if not hparams.MODEL.DETAIL.FIRST_ESP_ONLY:
            self.global_net = GlobalNet(channel_settings, output_shape, num_points)

    def forward(self, x):
        res_out = self.resnet(x)
        if not hparams.MODEL.DETAIL.FIRST_ESP_ONLY:
            global_re, global_out = self.global_net(res_out)
            return global_out

class Predictor(nn.Module):
    def __init__(self, inplanes, input_shape, output_shape, num_class):
        super(Predictor, self).__init__()
        self.predict = self._make_predictor(inplanes, input_shape, output_shape, num_class)
        globalvars.early_predictors.append(self)
    
    def _make_predictor(self, inplanes, input_shape, output_shape, num_class):
        layers = []
        # lateral of globalNet
        layers.append(nn.Conv2d(inplanes, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(globalvars.BatchNorm2dImpl(256))
        layers.append(nn.ReLU(inplace=True))

        # predict of globalNet
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(globalvars.BatchNorm2dImpl(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))
        if hparams.MODEL.LEARNABLE_OFFSET.USE_IN_PREDICTOR:
            layers.append(OffsetBlock(output_shape[0], output_shape[1], num_class, num_class, 256))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.predict(x)

class GlobalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_settings = channel_settings
        output_shape_factor = 2 ** (len(channel_settings) - 1)
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class, output_shape_factor))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
                output_shape_factor /= 2
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, globalvars.BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(globalvars.BatchNorm2dImpl(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))
        layers.append(globalvars.BatchNorm2dImpl(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class, output_shape_factor):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(globalvars.BatchNorm2dImpl(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))
        if hparams.MODEL.LEARNABLE_OFFSET.USE_IN_PREDICTOR:
            layers.append(OffsetBlock(output_shape[0], output_shape[1], num_class, 256, 256))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs
