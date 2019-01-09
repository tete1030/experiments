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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import lib.datasets as datasets
from lib.models.common import StrictNaNReLU
from lib.utils.transforms import fliplr_pts, kpt_affine
from lib.utils.evaluation import accuracy, OffsetCycleAverageMeter, parse_map, generate_ans, generate_mpii_ans
from lib.utils.imutils import batch_resize
from lib.models.displacechan import DisplaceChannel
from lib.models.spacenorm import SpaceNormalization
from utils.globals import config, hparams, globalvars
from utils.log import log_i
from utils.train import adjust_learning_rate
from utils.checkpoint import save_pred, load_pretrained_loose, save_checkpoint, RejectLoadError
from utils.miscs import nprand_init
from experiments.baseexperiment import BaseExperiment, EpochContext
from .augtrans import transform_maps

FACTOR = 4

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, num_groups=32):
        assert num_features % num_groups == 0, "num_features({}) is not dividend by num_groups({})".format(num_features, num_groups)
        super(GroupNormWrapper, self).__init__(num_groups, num_features, eps=1e-5)

class Experiment(BaseExperiment):
    def init(self):
        globalvars.offsetblock_counter = 0
        globalvars.displace_mods = list()
        globalvars.offsetblock_output = dict()

        self.data_source = hparams.DATASET.PROFILE
        if self.data_source == "coco":
            self.num_parts = datasets.mscoco.NUM_PARTS
            self.flip_index = datasets.mscoco.FLIP_INDEX
        elif self.data_source == "mpii":
            self.num_parts = datasets.mpii.NUM_PARTS
            self.flip_index = datasets.mpii.FLIP_INDEX
        else:
            assert False

        if hparams.MODEL.USE_GN:
            globalvars.BatchNorm2dImpl = GroupNormWrapper
        else:
            globalvars.BatchNorm2dImpl = nn.BatchNorm2d

        self.model = nn.DataParallel(MyPose(self.num_parts).cuda())
        assert globalvars.offsetblock_counter == len(hparams.MODEL.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO) or not hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK

        # Separate parameters
        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in globalvars.displace_mods if hasattr(dm, "offset")]))
            self.offset_regressor_parameters = list(filter(lambda x: x.requires_grad, list(itertools.chain.from_iterable([dm.offset_regressor.parameters() for dm in globalvars.displace_mods if hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET]))))
            if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORM_OFFSET:
                self.offset_transformer_parameters = list(filter(lambda x: x.requires_grad,
                    list(itertools.chain.from_iterable([dm.offset_transformer.parameters() for dm in globalvars.displace_mods])) + \
                    list(self.model.module.transformer.parameters())))
            else:
                self.offset_transformer_parameters = []
        else:
            self.offset_parameters = []
            self.offset_regressor_parameters = []
            self.offset_transformer_parameters = []

        special_parameter_ids = list(map(lambda x: id(x),
            self.offset_parameters + self.offset_regressor_parameters + self.offset_transformer_parameters))
        self.normal_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))

        # Initialize optimizers
        # Normal optimizer
        self.optimizer = torch.optim.Adam(
            self.normal_parameters,
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

        self.cur_lr = hparams.TRAIN.LEARNING_RATE

        if self.data_source == "coco":
            self.init_mscoco()
        elif self.data_source == "mpii":
            self.init_mpii()

        if hparams.DATASET.SUBSET is not None:
            if self.train_dataset:
                self.train_dataset = Subset(self.train_dataset, list(range(int(len(self.train_dataset) * hparams.DATASET.SUBSET))))

        self.worker_init_fn = nprand_init

        if self.offset_optimizer is not None:
            self.move_dis_avgmeter = []
            for dm in globalvars.displace_mods:
                if dm.offset.size(0) == 0:
                    continue
                self.move_dis_avgmeter.append(OffsetCycleAverageMeter(hparams.LOG.MOVE_AVERAGE_CYCLE, (dm.offset.data * dm.offset_scale).cpu()))
        else:
            self.move_dis_avgmeter = None

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
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNEL,
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
            kpmap_res=hparams.MODEL.OUT_SHAPE,
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
        if self.offset_optimizer:
            self.move_dis_avgmeter = checkpoint["move_dis_avgmeter"]
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_full, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict() if self.offset_optimizer else None,
            "transformer_optimizer": self.transformer_optimizer.state_dict() if self.transformer_optimizer else None,
            "move_dis_avgmeter": self.move_dis_avgmeter if not hparams.MODEL.DETAIL.DISABLE_DISPLACE else None
        }
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
                    tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, stat_type), {stat_name: stats[istat]}, step)

        elif self.data_source == "mpii":
            annotates = epoch_ctx.stored["annotates"]
            acc = accuracy(annotates["pred"], annotates["gt"], annotates["head_box"])
            if tb_writer:
                tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh"), {"avg": float(acc[0])}, step)
            results = list()
            results.append("avg: {:2.2f}".format(float(acc[0]) * 100))
            for i in range(0, acc.size(0)-1):
                if tb_writer:
                    tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "PCKh"), {datasets.mpii.PART_LABELS[i]: float(acc[i+1])}, step)
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
            self.cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams.TRAIN.LEARNING_RATE, hparams.TRAIN.SCHEDULE, hparams.TRAIN.LR_GAMMA) 
            if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
                self.set_offset_learning_rate(epoch, step)

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
        if self.transformer_optimizer:
            self.transformer_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if optimize_offset:
            self.offset_optimizer.step()
        if self.transformer_optimizer:
            self.transformer_optimizer.step()

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
            globalvars.main_context.tb_writer.add_scalars("{}/{}".format(hparams.LOG.TB_DOMAIN, "move_dis"), {"mod": np.mean(move_dis_avg), "mod_cur": np.mean(move_dis)}, progress["step"] + 1)

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer is not None and (progress["step"] + 1) % hparams.LOG.OFFSET_SAVE_INTERVAL == 0:
            self.save_offsets(progress["step"] + 1)

    def summarize_gradient(self):
        # TODO: all gradients? offset gradients?
        pass

    def summarize_parameter(self):
        # TODO: summarize all parameters
        pass

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

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer and progress["step"] == hparams.TRAIN.OFFSET.TRAIN_MIN_STEP and progress["train"]:
            self.save_offsets(progress["step"])

        det_map_gt_cuda = det_maps_gt.cuda()
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img
        output_maps, *offoutputs = self.model(img)

        mask_notlabeled = (keypoint[:, :, 2] <= 0.1).cuda()
        mask_labeled = (~mask_notlabeled)
        mask_visible = (keypoint[:, :, 2] > 1.1).cuda()
        mask_notvisible = (mask_labeled & (~mask_visible))

        if hparams.MODEL.DETAIL.LOSS_FINAL == "all":
            masking_final = 1.
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "labeled":
            masking_final = mask_labeled.float().view(-1, self.num_parts, 1, 1)
        elif hparams.MODEL.DETAIL.LOSS_FINAL == "visible":
            masking_final = mask_visible.float().view(-1, self.num_parts, 1, 1)
        else:
            assert False

        loss = ((output_maps - det_map_gt_cuda).pow(2) * \
            masking_final).mean().sqrt()

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE and self.offset_optimizer and self.transformer_optimizer and progress["step"] >= hparams.TRAIN.OFFSET.TRAIN_MIN_STEP and progress["train"] and hparams.MODEL.LOSS_FEATSTAB:
            scale = torch.tensor(truncnorm.rvs(-1, 1, loc=1, scale=0.5, size=batch_size)).float()
            rotate = torch.tensor(truncnorm.rvs(-1, 1, loc=0, scale=np.pi/6, size=batch_size)).float()
            # blur_sigma = torch.tensor(np.abs(truncnorm.rvs(-1, 1, loc=0, scale=3, size=batch_size))).float()
            mean_img = torch.tensor(self.val_dataset.mean, dtype=torch.float, device=img.device)[None, :, None, None]
            img_trans = transform_maps(img + mean_img, scale, rotate, None) - mean_img
            offoutputs_trans = list()
            for offout in offoutputs:
                offoutputs_trans.append(transform_maps(offout.detach(), scale, rotate, None))

            if config.vis and False:
                import matplotlib.pyplot as plt
                show_img_num = min(3, len(img))
                fig, axes = plt.subplots(show_img_num, 2, figsize=(16, 10 * show_img_num))
                img_show = self.val_dataset.restore_image(img.cpu())
                img_trans_show = self.val_dataset.restore_image(img_trans.cpu())
                for iimg in range(show_img_num):
                    axes[iimg, 0].imshow(img_show[iimg])
                    axes[iimg, 1].imshow(img_trans_show[iimg])
                plt.show()

            set_requires_grad(self.normal_parameters, False)
            set_requires_grad(self.offset_parameters, False)
            set_requires_grad(self.offset_regressor_parameters, False)
            _, *offoutputs_img_trans = self.model(img_trans)
            set_requires_grad(self.normal_parameters, True)
            set_requires_grad(self.offset_parameters, True)
            set_requires_grad(self.offset_regressor_parameters, True)

            feature_loss = 0
            for ioff, offout_trans in enumerate(offoutputs_trans):
                feature_loss = feature_loss + (offoutputs_img_trans[ioff] - offout_trans).pow(2).mean()

            loss = loss + hparams.MODEL.LOSS_FEATSTAB_COF * feature_loss
            epoch_ctx.add_scalar("feature_loss", feature_loss.item())

        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        epoch_ctx.add_scalar("loss", loss.item())

        if (loss.data != loss.data).any():
            import ipdb; ipdb.set_trace()

        if not is_train or config.vis:
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

        result = {
            "loss": loss,
            "index": batch["index"]
        }

        return result

def set_requires_grad(paras, requires_grad):
    for para in paras:
        para.requires_grad = requires_grad

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
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
                nn.Softplus(),
                SpaceNormalization())
        else:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
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
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1):
        super(OffsetBlock, self).__init__()
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
            self.displace_planes, self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=hparams.MODEL.DETAIL.DISPLACE_LEARNABLE_OFFSET,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transform_offset=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORM_OFFSET,
            num_transformer_channels=256,
            half_reversed_offset=hparams.MODEL.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=globalvars.displace_mods[-1] if hparams.MODEL.LEARNABLE_OFFSET.REUSE_OFFSET and len(globalvars.displace_mods) > 0 else None)
        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if hparams.MODEL.LEARNABLE_OFFSET.ENABLE_ATTEN:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=False, stride=stride)
        else:
            self.atten_displace = None
        if hparams.MODEL.LEARNABLE_OFFSET.ENABLE_MASK:
            self.atten_post = Attention(0, self.outplanes, input_shape=(self.out_height, self.out_width), bias_planes=inplanes // 4, bias_factor=2, space_norm=False)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x, transformer_source=None):
        if globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        out_pre = self.pre_offset(x)
        out_dis = self.displace(out_pre, transformer_source=transformer_source)

        if self.atten_displace is not None:
            out_atten = self.atten_displace(x)
        else:
            out_atten = None
        out_post = self.post_offset(out_atten * out_dis if out_atten is not None else out_dis)
        if self.downsample is not None:
            x = self.downsample(x)
        out_skip = x + (out_post * self.atten_post(x) if self.atten_post is not None else out_post)

        out_final = self.relu(self.bn(out_skip))

        device = out_post.device
        if out_post.device not in globalvars.offsetblock_output:
            globalvars.offsetblock_output[device] = list()
        globalvars.offsetblock_output[device].append(out_post)

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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = globalvars.BatchNorm2dImpl(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = globalvars.BatchNorm2dImpl(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = globalvars.BatchNorm2dImpl(planes * 4)
        self.relu = StrictNaNReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

    def forward(self, x):
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

class MyPose(nn.Module):
    def __init__(self, num_class):
        super(MyPose, self).__init__()
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORM_OFFSET:
            self.transformer = TransformFeature()
        self.estimator = SimpleEstimator(num_class)

    def forward(self, x):
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORM_OFFSET:
            transform_features = self.transformer(x)
        else:
            transform_features = None
        prediction = self.estimator(x, transform_features)
        if len(globalvars.offsetblock_output) > 0:
            block_outputs = globalvars.offsetblock_output[x.device]
            globalvars.offsetblock_output[x.device] = list()
        else:
            block_outputs = []
        return (prediction, *block_outputs)

class TransformFeature(nn.Module):
    def __init__(self):
        super(TransformFeature, self).__init__()
        self.inplanes = 64
        self.inshape_factor = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.inshape_factor *= 2
        self.bn1 = globalvars.BatchNorm2dImpl(64)
        self.relu = StrictNaNReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inshape_factor *= 2
        self.layer1 = self._make_layer(Bottleneck, 64, 3, res_index=0)

    def _make_layer(self, block, planes, blocks, res_index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                globalvars.BatchNorm2dImpl(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=0, stride=stride, downsample=downsample))
        if stride != 1:
            self.inshape_factor *= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=i))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)

        return x1

class MySequential(nn.Sequential):
    def forward(self, input, extra):
        for module in self._modules.values():
            if isinstance(module, OffsetBlock):
                input = module(input, extra)
            else:
                input = module(input)
        return input

class SimpleEstimator(nn.Module):
    def __init__(self, num_class):
        super(SimpleEstimator, self).__init__()
        self.inplanes = 64
        self.inshape_factor = 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.inshape_factor *= 2
        self.bn1 = globalvars.BatchNorm2dImpl(64)
        self.relu = StrictNaNReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inshape_factor *= 2
        self.layer1 = self._make_layer(Bottleneck, 64, 3, res_index=0)

        self.predictor = self._make_predictor(256, num_class)

    def _add_offset_block(self, layers):
        if hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            layers.append(OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.inplanes,
                int(self.inplanes * hparams.MODEL.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO[globalvars.offsetblock_counter])))
            globalvars.offsetblock_counter += 1

    def _make_layer(self, block, planes, blocks, res_index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                globalvars.BatchNorm2dImpl(planes * block.expansion),
            )

        layers = []

        self._add_offset_block(layers)
        layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=0, stride=stride, downsample=downsample))
        if stride != 1:
            self.inshape_factor *= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            self._add_offset_block(layers)
            layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=i))

        return MySequential(*layers)

    def _make_predictor(self, planes, num_class):
        layers = []
        layers.append(nn.Conv2d(planes, planes,
            kernel_size=1, stride=1, bias=False))
        layers.append(globalvars.BatchNorm2dImpl(planes))
        layers.append(StrictNaNReLU(inplace=True))

        layers.append(nn.Conv2d(planes, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x, transform_features):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x, transform_features)

        return self.predictor(x1)
