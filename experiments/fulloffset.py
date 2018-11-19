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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import lib.models as models
import lib.datasets as datasets
from lib.models.common import StrictNaNReLU
from lib.models.displacechan import DisplaceChannel
from lib.utils.transforms import fliplr_pts
from lib.utils.evaluation import AverageMeter, CycleAverageMeter, accuracy
from utils.globals import config, hparams, globalvars
from utils.log import log_i, log_w, log_progress
from utils.train import adjust_learning_rate, TrainContext, ValidContext
from utils.checkpoint import save_pred, load_pretrained_loose, save_checkpoint, RejectLoadError
from lib.utils.lambdalayer import Lambda
from experiments.baseexperiment import BaseExperiment, EpochContext

FACTOR = 4

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, num_groups=32):
        assert num_features % num_groups == 0, "num_features({}) is not dividend by num_groups({})".format(num_features, num_groups)
        super(GroupNormWrapper, self).__init__(num_groups, num_features, eps=1e-5)

BatchNorm2dImpl = None

class Experiment(BaseExperiment):
    exp = None
    def init(self):
        if Experiment.exp is not None:
            log_w("Singleton experiment is already taken")
        Experiment.exp = self
        self.early_predictor_size = list()
        self.displace_mods = list()
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            self.early_predictors = list()
            self.pre_early_predictor_outs = dict()

        self.data_source = hparams.DATASET.DATA
        if self.data_source == "coco":
            self.num_parts = datasets.mscoco.NUM_PARTS
            self.flip_index = datasets.mscoco.FLIP_INDEX
        elif self.data_source == "mpii":
            self.num_parts = datasets.mpii.NUM_PARTS
            self.flip_index = datasets.mpii.FLIP_INDEX
        else:
            assert False

        pretrained = hparams.MODEL.RESNET_PRETRAINED
        if config.resume:
            pretrained = None

        global BatchNorm2dImpl
        if hparams.MODEL.USE_GN:
            BatchNorm2dImpl = GroupNormWrapper
        else:
            BatchNorm2dImpl = nn.BatchNorm2d

        self.model = nn.DataParallel(Controller(MainModel(hparams.MODEL.OUT_SHAPE[::-1], self.num_parts, pretrained=pretrained)).cuda())
        assert OffsetBlock._counter == len(hparams.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO) or not hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in self.displace_mods if hasattr(dm, "offset")]))
            self.offset_regressor_parameters = list(filter(lambda x: x.requires_grad, list(itertools.chain.from_iterable([dm.offset_regressor.parameters() for dm in self.displace_mods if hasattr(dm, "offset_regressor")]))))
        else:
            self.offset_parameters = []
            self.offset_regressor_parameters = []

        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            self.early_predictor_parameters = list(filter(lambda x: x.requires_grad, itertools.chain.from_iterable([ep.parameters() for ep in self.early_predictors])))
        else:
            self.early_predictor_parameters = []

        special_parameter_ids = list(map(lambda x: id(x), self.offset_parameters + self.offset_regressor_parameters + self.early_predictor_parameters))
        self.normal_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))

        self.optimizer = torch.optim.Adam(
            self.normal_parameters,
            lr=hparams.LEARNING_RATE,
            weight_decay=hparams.WEIGHT_DECAY)

        offset_optimizer_args = []
        if len(self.offset_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_lr", "params": self.offset_parameters, "lr": hparams.LEARNABLE_OFFSET.LR, "init_lr": hparams.LEARNABLE_OFFSET.LR})
        if len(self.offset_regressor_parameters) > 0:
            offset_optimizer_args.append(
                {"para_name": "offset_regressor_lr", "params": self.offset_regressor_parameters, "lr": hparams.LEARNABLE_OFFSET.LR_REGRESSOR, "init_lr": hparams.LEARNABLE_OFFSET.LR_REGRESSOR})
        if len(offset_optimizer_args) > 0:
            self.offset_optimizer = torch.optim.Adam(offset_optimizer_args)
        else:
            self.offset_optimizer = None

        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            self.early_predictor_optimizer = torch.optim.Adam(
                self.early_predictor_parameters,
                lr=hparams.LEARNING_RATE,
                weight_decay=hparams.WEIGHT_DECAY)
        else:
            self.early_predictor_optimizer = None

        self.criterion = nn.MSELoss()
        
        self.cur_lr = hparams.LEARNING_RATE

        if self.data_source == "coco":
            self.coco = COCO("data/mscoco/person_keypoints_train2017.json")
            self.train_dataset = datasets.COCOSinglePose("data/mscoco/images2017",
                                                self.coco,
                                                "data/mscoco/sp_split_2017.pth",
                                                "data/mscoco/" + hparams.DATASET.MEAN_STD_FILE,
                                                True,
                                                img_res=hparams.MODEL.INP_SHAPE,
                                                ext_border=hparams.DATASET.EXT_BORDER,
                                                kpmap_res=hparams.MODEL.OUT_SHAPE,
                                                keypoint_res=hparams.MODEL.OUT_SHAPE,
                                                kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
                                                scale_factor=hparams.DATASET.SCALE_FACTOR,
                                                rot_factor=hparams.DATASET.ROTATE_FACTOR,
                                                trans_factor=hparams.DATASET.TRANSLATION_FACTOR)

            self.val_dataset = datasets.COCOSinglePose("data/mscoco/images2017",
                                                self.coco,
                                                "data/mscoco/sp_split_2017.pth",
                                                "data/mscoco/" + hparams.DATASET.MEAN_STD_FILE,
                                                False,
                                                img_res=hparams.MODEL.INP_SHAPE,
                                                ext_border=hparams.DATASET.EXT_BORDER,
                                                kpmap_res=hparams.MODEL.OUT_SHAPE,
                                                keypoint_res=hparams.MODEL.OUT_SHAPE,
                                                kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
                                                scale_factor=hparams.DATASET.SCALE_FACTOR,
                                                rot_factor=hparams.DATASET.ROTATE_FACTOR,
                                                trans_factor=hparams.DATASET.TRANSLATION_FACTOR)
            self.train_collate_fn = datasets.COCOSinglePose.collate_function
            self.valid_collate_fn = datasets.COCOSinglePose.collate_function
        elif self.data_source == "mpii":
            self.train_dataset = datasets.MPII("data/mpii/images",
                "data/mpii/mpii_human_pose.json",
                "data/mpii/split_sig.pth",
                "data/mpii/mean_std.pth",
                True,
                True,
                img_res=hparams.MODEL.INP_SHAPE,
                kpmap_res=hparams.MODEL.OUT_SHAPE,
                kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
                scale_factor=hparams.DATASET.SCALE_FACTOR,
                rot_factor=hparams.DATASET.ROTATE_FACTOR,
                trans_factor=hparams.DATASET.MPII.TRANSLATION_FACTOR)

            self.val_dataset = datasets.MPII("data/mpii/images",
                "data/mpii/mpii_human_pose.json",
                "data/mpii/split_sig.pth",
                "data/mpii/mean_std.pth",
                False,
                True,
                img_res=hparams.MODEL.INP_SHAPE,
                kpmap_res=hparams.MODEL.OUT_SHAPE,
                kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
                scale_factor=hparams.DATASET.SCALE_FACTOR,
                rot_factor=hparams.DATASET.ROTATE_FACTOR,
                trans_factor=hparams.DATASET.MPII.TRANSLATION_FACTOR)
            self.train_collate_fn = datasets.MPII.collate_function
            self.valid_collate_fn = datasets.MPII.collate_function

        self.worker_init_fn = datasets.mscoco.worker_init
        self.print_iter_start = " | "

        if self.offset_optimizer is not None:
            self.move_dis_avgmeter = []
            for dm in self.displace_mods:
                if dm.offset.size(0) == 0:
                    continue
                self.move_dis_avgmeter.append(Experiment.OffsetCycleAverageMeter(hparams.LEARNABLE_OFFSET.MOVE_AVERAGE_CYCLE, (dm.offset.data * dm.scale).cpu()))
        else:
            self.move_dis_avgmeter = None

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
        if not no_criterion_load:
            self.criterion.load_state_dict(checkpoint["criterion"])
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
            "criterion": self.criterion.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict() if self.offset_optimizer else None,
            "early_predictor_optimizer": self.early_predictor_optimizer.state_dict() if self.early_predictor_optimizer else None,
            "move_dis_avgmeter": self.move_dis_avgmeter if not hparams.MODEL.DETAIL.DISABLE_DISPLACE else None
        }
        save_checkpoint(checkpoint_dict, checkpoint_full=checkpoint_full, force_replace=True)

    @staticmethod
    def _summarize_tensorboard(eval_result, params, step):
        def _summarize(ap, iou_thr=None, area_rng="all", max_dets=100, title=None):
            type_str = "AP" if ap==1 else "AR"
            if title is None:
                iou_str = "{:0.2f}-{:0.2f}".format(params.iouThrs[0], params.iouThrs[-1]) \
                    if iou_thr is None else "{:0.2f}".format(iou_thr)
                title = "{:<9}_{:>6s}_{:>3d}".format(iou_str, area_rng, max_dets)

            aind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == area_rng]
            mind = [i for i, mDet in enumerate(params.maxDets) if mDet == max_dets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval_result["precision"]
                # IoU
                if iou_thr is not None:
                    t = np.where(iou_thr == params.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = eval_result["recall"]
                if iou_thr is not None:
                    t = np.where(iou_thr == params.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, type_str), {title: mean_s}, step)
            return mean_s

        _summarize(1, title="avg", max_dets=20)
        _summarize(1, title="i50", max_dets=20, iou_thr=.5)
        _summarize(1, title="i75", max_dets=20, iou_thr=.75)
        _summarize(1, title="med", max_dets=20, area_rng="medium")
        _summarize(1, title="lar", max_dets=20, area_rng="large")
        _summarize(0, title="avg", max_dets=20)
        _summarize(0, title="i50", max_dets=20, iou_thr=.5)
        _summarize(0, title="i75", max_dets=20, iou_thr=.75)
        _summarize(0, title="med", max_dets=20, area_rng="medium")
        _summarize(0, title="lar", max_dets=20, area_rng="large")

    def evaluate(self, epoch_ctx:EpochContext, epoch, step):
        if "annotates" not in epoch_ctx.stored:
            return
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

                self._summarize_tensorboard(coco_eval.eval, coco_eval.params, step)
                coco_eval.summarize()
            else:
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AP"), {"avg": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AP"), {"i50": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AP"), {"i75": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AP"), {"med": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AP"), {"lar": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AR"), {"avg": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AR"), {"i50": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AR"), {"i75": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AR"), {"med": 0}, step)
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "AR"), {"lar": 0}, step)

                print("No points")
        elif self.data_source == "mpii":
            annotates = epoch_ctx.stored["annotates"]
            acc = accuracy(annotates["pred"], annotates["gt"], annotates["head_box"])
            globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh"), {"avg": float(acc[0])}, step)
            results = list()
            results.append("avg: {:2.2f}".format(float(acc[0]) * 100))
            for i in range(0, acc.size(0)-1):
                globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh"), {datasets.mpii.PART_LABELS[i]: float(acc[i+1])}, step)
                results.append("{}: {:2.2f}".format(datasets.mpii.PART_LABELS[i], float(acc[i+1]) * 100))
            print(" | ".join(results) + "\n")

    def process_stored(self, epoch_ctx:EpochContext, epoch, step):
        if config.store:
            for store_key in epoch_ctx.stored:
                if epoch == 0:
                    pred_file = "{}_evaluate.npy".format(store_key)
                else:
                    pred_file = "{}_{}.npy".format(store_key, epoch)
                save_pred(epoch_ctx.stored[store_key], checkpoint_folder=config.checkpoint_dir, pred_file=pred_file)

    def set_offset_learning_rate(self, epoch, step):
        if self.offset_optimizer is None:
            return

        if step >= hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP and hparams.LEARNABLE_OFFSET.LR_DECAY_STEP > 0 and hparams.LEARNABLE_OFFSET.LR_GAMMA > 0:
            step_offset = max(0, step - hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP)
        else:
            step_offset = -1

        for param_group in self.offset_optimizer.param_groups:
            if step_offset >= 0:
                cur_lr_offset = param_group["init_lr"] * (hparams.LEARNABLE_OFFSET.LR_GAMMA ** (float(step_offset) / hparams.LEARNABLE_OFFSET.LR_DECAY_STEP))
                log_i("Set {} to {:.5f}".format(param_group["para_name"], cur_lr_offset))
            else:
                cur_lr_offset = param_group["init_lr"]
            param_group["lr"] = cur_lr_offset

    def set_offset_learning_para(self, epoch, step):
        for dm in self.displace_mods:
            if dm.LO_interpolate_kernel_type == "gaussian" and dm.learnable_offset and dm.LO_active:
                if step >= hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP and hparams.LEARNABLE_OFFSET.INTERPOLATE_GAUSSIAN_SIGMA_DECAY_STEP > 0 and hparams.LEARNABLE_OFFSET.INTERPOLATE_GAUSSIAN_SIGMA_DECAY_RATE > 0:
                    step_offset = max(0, step - hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP)
                    LO_sigma_new = float(dm.LO_sigma_init) * (hparams.LEARNABLE_OFFSET.INTERPOLATE_GAUSSIAN_SIGMA_DECAY_RATE ** (float(step_offset) / hparams.LEARNABLE_OFFSET.INTERPOLATE_GAUSSIAN_SIGMA_DECAY_STEP))
                    LO_kernel_size_new = int(LO_sigma_new * 3) * 2 + 1
                    dm.set_learnable_offset_para(LO_kernel_size_new, LO_sigma_new)

                if dm.LO_kernel_size == 1:
                    dm.switch_LO_state(False)

    def epoch_start(self, epoch, step, evaluate_only):
        if not evaluate_only:
            self.cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams.LEARNING_RATE, hparams.SCHEDULE, hparams.LR_GAMMA)
            adjust_learning_rate(self.early_predictor_optimizer, epoch, hparams.LEARNING_RATE, hparams.SCHEDULE, hparams.LR_GAMMA)
            if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
                self.set_offset_learning_rate(epoch, step)
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.set_offset_learning_para(epoch, step)

    class OffsetCycleAverageMeter(object):
        """Computes and stores the cycled average of offset"""
        def __init__(self, size, first):
            self.size = size
            self.reset()
            self.update(first)

        def reset(self):
            self._pool = list()
            self._pointer = 0
            self.count = 0
            # self.val = None
            self.avg = None
            self.lastdiff = None

        def update(self, val, n=1):
            for i in range(n):
                if self.count >= self.size:
                    self._pool[self._pointer] = val
                    self._pointer = (self._pointer + 1) % self.size
                else:
                    self._pool.append(val)
                    self._pointer = (self._pointer + 1) % self.size
                    self.count += 1

            if self.count > 1:
                self.lastdiff = (val - self._pool[(self._pointer + self.count - 2) % self.count]).abs().mean().item()
                self.avg = ((val - self._pool[self._pointer % self.count]) / (self.count - 1)).abs().mean().item()
            else:
                self.lastdiff = None
                self.avg = None

    @staticmethod
    def set_para_require_grad(paras, requires_grad):
        for para in paras:
            para.requires_grad = requires_grad

    def save_offsets(self, step):
        offset_disabled = True
        for dm in self.displace_mods:
            if dm.LO_active:
                offset_disabled = False
        if not offset_disabled:
            torch.save([(dm.get_all_offsets(detach=True) * dm.scale).cpu() for dm in self.displace_mods], os.path.join(config.checkpoint_dir, "offset_{}.pth".format(step)))

    def epoch_end(self, epoch, step, evaluate_only):
        if not evaluate_only and not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.save_offsets(step)

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        optimize_offset = False
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer is not None and progress["step"] >= hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP:
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
            for idm in range(len(self.displace_mods)):
                dm = self.displace_mods[idm]
                if dm.offset.size(0) == 0:
                    continue
                self.move_dis_avgmeter[idm].update((dm.offset.detach() * dm.scale).cpu())
                move_dis_avg.append(self.move_dis_avgmeter[idm].avg)
                move_dis.append(self.move_dis_avgmeter[idm].lastdiff)
            globalvars.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "move_dis"), {"mod": np.mean(move_dis_avg), "mod_cur": np.mean(move_dis)}, progress["step"] + 1)

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer is not None and (progress["step"] + 1) % hparams.LEARNABLE_OFFSET.OFFSET_SAVE_INTERVAL == 0:
            self.save_offsets(progress["step"] + 1)

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, progress: dict) -> dict:
        image_ids = batch["img_index"].tolist()
        img = batch["img"]
        det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        img_ori_size = batch["img_ori_size"]
        keypoint = batch["keypoint"]
        is_train = progress["train"]
        batch_size = img.size(0)
        globalvars.progress = progress

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer and progress["step"] == hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP and progress["train"]:
            self.save_offsets(progress["step"])

        det_map_gt_cuda = [dm.cuda() for dm in det_maps_gt]
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img
        output_maps, early_predictor_outputs = self.model(img)
        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        loss = 0.

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

        epoch_ctx.add_scalar("loss", loss.item())

        if (loss.data != loss.data).any():
            import ipdb; ipdb.set_trace()

        if not is_train or config.vis:
            kp_pred, score = parse_map(output_maps[-1], thres=hparams.MODEL.PARSE_THRESHOLD)
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

            if config.store and hparams.CONFIG.STORE_MAP and is_train:
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

        result = {
            "loss": loss,
            "index": batch["index"]
        }

        return result

# Attention List:
# - middle_outputs
# - BreakableSequential in Blocks
# - 

class Controller(nn.Module):
    def __init__(self, main_model):
        super(Controller, self).__init__()
        self.main_model = main_model
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            early_predictor = list()
            assert len(Experiment.exp.early_predictor_size) > 0
            for inplanes, inshape_factor in Experiment.exp.early_predictor_size:
                early_predictor.append(
                    Predictor(
                        inplanes, 
                        (hparams.MODEL.INP_SHAPE[1] // inshape_factor, hparams.MODEL.INP_SHAPE[0] // inshape_factor),
                        hparams.MODEL.OUT_SHAPE[::-1],
                        Experiment.exp.num_parts))
            self.early_predictor = nn.ModuleList(early_predictor)
        else:
            self.early_predictor = None

    def forward(self, x):
        if self.early_predictor:
            Experiment.exp.pre_early_predictor_outs[x.device] = list()

        out = self.main_model(x)

        if self.early_predictor:
            pre_early_predictor_outs = Experiment.exp.pre_early_predictor_outs[x.device]
            Experiment.exp.pre_early_predictor_outs[x.device] = list()
            assert len(pre_early_predictor_outs) == len(self.early_predictor)
            return out, [self.early_predictor[i](pre_early_predictor_outs[i]) for i in range(len(pre_early_predictor_outs))]
        else:
            return out, None
        
class MainModel(nn.Module):
    def __init__(self, output_shape, num_points, pretrained=None):
        super(MainModel, self).__init__()
        if hparams.MODEL.RESNET == 18:
            self.resnet = resnet18(pretrained=pretrained)
            self.global_net = GlobalNet([512, 256, 128, 64], output_shape, num_points)
        elif hparams.MODEL.RESNET == 50:
            self.resnet = resnet50(pretrained=pretrained)
            self.global_net = GlobalNet([2048, 1024, 512, 256], output_shape, num_points)
        elif hparams.MODEL.RESNET == 101:
            self.resnet = resnet101(pretrained=pretrained)
            self.global_net = GlobalNet([2048, 1024, 512, 256], output_shape, num_points)
        else:
            assert False

    def forward(self, x):
        res_out = self.resnet(x)
        global_re, global_out = self.global_net(res_out)
        return global_out

class Predictor(nn.Module):
    def __init__(self, inplanes, input_shape, output_shape, num_class):
        super(Predictor, self).__init__()
        self.predict = self._make_predictor(inplanes, input_shape, output_shape, num_class)
        Experiment.exp.early_predictors.append(self)
    
    def _make_predictor(self, inplanes, input_shape, output_shape, num_class):
        layers = []
        # lateral of globalNet
        layers.append(nn.Conv2d(inplanes, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        # predict of globalNet
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))
        if hparams.LEARNABLE_OFFSET.USE_IN_PREDICTOR:
            layers.append(OffsetBlock(output_shape[0], output_shape[1], num_class, 256))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.predict(x)

class SpaceNormalization(nn.Module):
    def __init__(self):
        super(SpaceNormalization, self).__init__()

    def forward(self, x):
        x = x + torch.tensor(np.finfo(np.float32).eps, device=x.device, dtype=torch.float)
        return x / x.sum(-1, keepdim=True).sum(-2, keepdim=True)

class GaussianBlur(nn.Module):
    def __init__(self, inplanes, kernel_size=3, sigma=1):
        super(GaussianBlur, self).__init__()
        kernel_size = _pair(kernel_size)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        y = torch.arange(-int(kernel_size[0] // 2), int(kernel_size[0] // 2) + 1)
        x = torch.arange(-int(kernel_size[1] // 2), int(kernel_size[1] // 2) + 1)
        field = torch.stack([x.expand(kernel_size[0], -1), y[:, None].expand(-1, kernel_size[1])], dim=2).float()
        self.inplanes = inplanes
        self.kernel_size = kernel_size
        self.register_buffer("gaussian_kernel", torch.exp(- field.pow(2).sum(dim=2) / 2 / float(sigma) ** 2).view(1, 1, kernel_size[0], kernel_size[1]).repeat(self.inplanes, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.gaussian_kernel, padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2), groups=self.inplanes)

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
                nn.BatchNorm2d(outplanes, momentum=hparams.LEARNABLE_OFFSET.BN_MOMENTUM),
                nn.Softplus(),
                SpaceNormalization())
        else:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.LEARNABLE_OFFSET.BN_MOMENTUM),
                nn.Sigmoid())

    def forward(self, x):
        if self.bias is not None:
            atten_bias = nn.functional.interpolate(self.bias, size=x.size()[-2:], mode="bilinear", align_corners=True).expand(x.size(0), -1, -1, -1)
            if self.inplanes > 0:
                x = torch.cat([x, atten_bias], dim=1)
            else:
                x = atten_bias

        return self.atten(x)

class OffsetBlock(nn.Module):
    _counter = 0
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1):
        super(OffsetBlock, self).__init__()
        LO_interpolate_kernel_type = hparams.LEARNABLE_OFFSET.INTERPOLATE_KERNEL_TYPE
        if LO_interpolate_kernel_type == "gaussian":
            LO_sigma = hparams.LEARNABLE_OFFSET.INTERPOLATE_GAUSSIAN_SIGMA
            LO_kernel_size = int(LO_sigma * 3) * 2 + 1
        else:
            LO_sigma = 0.
            LO_kernel_size = 3
        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.displace_planes = displace_planes
        self.outplanes = outplanes
        self.stride = stride
        self.displace = DisplaceChannel(
            self.out_height, self.out_width,
            1,
            self.displace_planes,
            learnable_offset=hparams.MODEL.DETAIL.DISPLACE_LEARNABLE_OFFSET,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            random_offset_init=hparams.MODEL.DETAIL.RANDOM_OFFSET_INIT,
            use_origin=True,
            actual_stride=1,
            displace_size=(1, 1),
            LO_interpolate_kernel_type=LO_interpolate_kernel_type,
            LO_kernel_size=LO_kernel_size,
            LO_sigma=LO_sigma,
            LO_balance_grad=False,
            free_offset_per_init_pos=int(self.displace_planes // hparams.LEARNABLE_OFFSET.BIND_CHAN),
            dconv_for_LO_stride=hparams.LEARNABLE_OFFSET.DCONV_FOR_LO_STRIDE,
            regress_offset=hparams.LEARNABLE_OFFSET.REGRESS_OFFSET,
            LO_half_reversed_offset=hparams.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=Experiment.exp.displace_mods[-1] if hparams.LEARNABLE_OFFSET.REUSE_OFFSET and len(Experiment.exp.displace_mods) > 0 else None)
        Experiment.exp.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if hparams.LEARNABLE_OFFSET.ENABLE_ATTEN:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=True, stride=stride)
        else:
            self.atten_displace = None
        if hparams.LEARNABLE_OFFSET.ENABLE_MASK:
            self.atten_post = Attention(0, self.outplanes, input_shape=(self.out_height, self.out_width), bias_planes=inplanes // 4, bias_factor=2, space_norm=False)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.LEARNABLE_OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        if config.check:
            assert x.size(2) == self.height and x.size(3) == self.width

        if globalvars.progress["step"] < hparams.LEARNABLE_OFFSET.TRAIN_MIN_STEP:
            return x

        out_pre = self.pre_offset(x)
        out_dis, out_dis_LO = self.displace(out_pre)
        if out_dis_LO is not None:
            out_dis = out_dis_LO
        if self.atten_displace is not None:
            out_atten = self.atten_displace(x)
        else:
            out_atten = None
        out_post = self.post_offset(out_atten * out_dis if out_atten is not None else out_dis)
        if self.downsample is not None:
            x = self.downsample(x)
        out_skip = x + (out_post * self.atten_post(x) if self.atten_post is not None else out_post)

        out_final = self.relu(self.bn(out_skip))

        if config.debug_nan:
            def get_backward_hook(var_name):
                def _backward_hook(grad):
                    exp = self
                    if isinstance(grad, torch.Tensor) and (grad.data != grad.data).any():
                        print("[OffsetBlock] " + var_name + " contains NaN during backward")
                        import ipdb; ipdb.set_trace()
                return _backward_hook

            all_var_names = ["x", "out_pre", "out_dis", "out_atten", "out_post", "out_skip", "out_final"]

            print("[OffsetBlock] !!!!!PERFORMANCE WARN: BACKWARD NAN DEBUGGING ENABLED!!!!!")
            for var_name in all_var_names:
                cur_var = locals()[var_name]
                if not (cur_var.data == cur_var.data).all():
                    print("[OffsetBlock] " + var_name + " contains NaN during forward")
                    import ipdb; ipdb.set_trace()
                cur_var.register_hook(get_backward_hook(var_name))

        return out_final

class ConvBlockWithAtten(nn.Module):
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1):
        super(ConvBlockWithAtten, self).__init__()
        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        # self.displace_planes = displace_planes
        self.displace_planes = inplanes // 2
        self.outplanes = outplanes
        self.stride = stride
        self.pre_poffset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_poffset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        self.atten_pdisplace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=True, stride=stride)
        self.pdisplace = nn.Conv2d(self.displace_planes, self.displace_planes, (3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.LEARNABLE_OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        dis = self.post_poffset(self.pdisplace(self.pre_poffset(x)) * self.atten_pdisplace(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(self.bn(x + dis))

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = BatchNorm2dImpl(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2dImpl(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dImpl(planes * 4)
        self.relu = StrictNaNReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if not (self.res_index in [1, 2, 3] and self.block_index == 1) and (self.res_index != 2 or self.block_index < 6) and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            self.offset_block = OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.inplanes,
                int(self.inplanes * hparams.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO[OffsetBlock._counter]))
            OffsetBlock._counter += 1
        else:
            self.offset_block = None

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        out = self.relu(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dImpl(planes)
        self.relu = StrictNaNReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dImpl(planes)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if not (self.res_index in [1, 2, 3] and self.block_index == 1) and (self.res_index != 2 or self.block_index < 6) and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            self.offset_block = OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.inplanes,
                int(self.inplanes * hparams.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO[OffsetBlock._counter]))
            OffsetBlock._counter += 1
        else:
            self.offset_block = None

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class BreakableSequential(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
            if input is None:
                return None
        return input

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        self.inshape_factor = 1
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.inshape_factor *= 2
        self.bn1 = BatchNorm2dImpl(64)
        self.relu = StrictNaNReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inshape_factor *= 2
        self.layer1 = self._make_layer(block, 64, layers[0], res_index=0)
        self.layer2 = self._make_layer(block, 128, layers[1], res_index=1, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], res_index=2, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], res_index=3, stride=2)

        for mod_name, m in self.named_modules():
            # TODO:
            if re.match(r"^(.+\.)?displace(\..+)?$", mod_name):
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, res_index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2dImpl(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=0, stride=stride, downsample=downsample))
        if stride != 1:
            self.inshape_factor *= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=i))

        if res_index < 3:
            Experiment.exp.early_predictor_size.append((self.inplanes, self.inshape_factor))

        # TODO:
        # Use BreakableSequential to support middle break 
        return BreakableSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # TODO:
        x1, x2, x3, x4 = None, None, None, None

        # TODO:
        x1 = self.layer1(x)
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            Experiment.exp.pre_early_predictor_outs[x.device].append(x1)
        if x1 is not None:
            x2 = self.layer2(x1)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
                Experiment.exp.pre_early_predictor_outs[x.device].append(x2)
        if x2 is not None:
            x3 = self.layer3(x2)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
                Experiment.exp.pre_early_predictor_outs[x.device].append(x3)
        if x3 is not None:
            x4 = self.layer4(x3)

        return [x4, x3, x2, x1]

def resnet18(pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet18 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model

def resnet50(pretrained=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet50 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model

def resnet101(pretrained=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet101 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model

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
            elif isinstance(m, BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))
        layers.append(BatchNorm2dImpl(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class, output_shape_factor):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))
        if hparams.LEARNABLE_OFFSET.USE_IN_PREDICTOR:
            layers.append(OffsetBlock(output_shape[0], output_shape[1], num_class, 256))

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


def parse_map(det_map, thres=0.1):
    det_map = det_map.detach()
    if det_map.is_cuda:
        det_map = det_map.cpu()
    det_map = det_map.numpy()
    num_batch = det_map.shape[0]
    num_part = det_map.shape[1]
    height = det_map.shape[2]
    width = det_map.shape[3]

    pred = np.zeros((num_batch, num_part, 3), dtype=np.float32)
    score = np.zeros((num_batch, num_part), dtype=np.float32)
    for sample_i in range(num_batch):
        for part_i in range(num_part):
            loc = det_map[sample_i, part_i].argmax().item()
            y = loc // width
            x = loc % width
            score_sp = det_map[sample_i, part_i, y, x]
            # TODO: test always 1 and always store score
            if score_sp > thres:
                pred[sample_i, part_i, 2] = 1
                score[sample_i, part_i] = score_sp
            if det_map[sample_i, part_i, y, max(0, x-1)] < det_map[sample_i, part_i, y, min(width-1, x+1)]:
                off_x = 0.25
            else:
                off_x = -0.25
            if det_map[sample_i, part_i, max(0, y-1), x] < det_map[sample_i, part_i, min(height-1, y+1), x]:
                off_y = 0.25
            else:
                off_y = -0.25
            pred[sample_i, part_i, 0] = x + 0.5 + off_x
            pred[sample_i, part_i, 1] = y + 0.5 + off_y
    return pred, score

def generate_ans(image_ids, preds, scores):
    ans = []
    for sample_i in range(len(preds)):
        image_id = image_ids[sample_i]

        val = preds[sample_i]
        score = scores[sample_i].mean()
        if val[:, 2].max()>0:
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(score)}
            # # p: average detected locations
            # p = val[val[:, 2] > 0][:, :2].mean(axis = 0)
            # for j in val:
            #     if j[2]>0.:
            #         tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
            #     else:
            #         # TRICK: for not detected points, place them at the average point
            #         tmp["keypoints"] += [float(p[0]), float(p[1]), 0]
            tmp["keypoints"] = val.ravel().tolist()
            ans.append(tmp)
    return ans

def generate_mpii_ans(image_ids, person_ids, preds):
    assert len(image_ids) == len(person_ids) and len(person_ids) == len(preds)
    return {"image_ids": image_ids, "person_ids": person_ids, "preds": preds}

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)
