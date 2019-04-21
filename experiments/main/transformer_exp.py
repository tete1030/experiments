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
from torch.utils.data import Subset, Dataset
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
from .offset import IndpendentTransformerRegressor
from lib.utils.augtrans import transform_maps, get_batch_transform_mats, batch_warp_affine
from scipy.stats import truncnorm
from lib.models.displacechan import TransformCoordinate
from lib.utils.transforms import get_transform

FACTOR = 4

class TransformerExperiment(BaseExperiment):
    def init(self):
        super().init()
        self.print_iter_start = "\n\t"

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

        self.model = nn.DataParallel(IndpendentTransformerRegressor(
            hparams.MODEL.IND_TRANSFORMER.NUM_FEATURE, hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_REGRESS, sep_scale=True).cuda())
        self.loss = TransformerLoss()

    def init_optimizer(self):
        def _print_parameters(para_groups, all_para_pair):
            print("Parameter groups:")
            para2name_dict = dict(map(lambda p: (p[1], p[0]), all_para_pair))
            for group_name, paras in para_groups.items():
                print("  " + group_name + ":")
                for para_name in map(para2name_dict.get, paras):
                    print("    " + para_name)
                print("")

        if not hparams.MODEL.DETAIL.DISABLE_DISPLACE:
            self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in globalvars.displace_mods if hasattr(dm, "offset")]))
        else:
            self.offset_parameters = []

        all_special_parameters = self.offset_parameters

        special_parameter_ids = list(map(lambda x: id(x), all_special_parameters))
        self.general_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))

        _print_parameters(
            dict(
                offset=self.offset_parameters),
            self.model.named_parameters())

        # Make sure no parameter is shared
        all_parameter_ptrs = list(map(lambda x: x.data_ptr(), all_special_parameters + self.general_parameters))
        assert len(all_parameter_ptrs) == len(np.unique(all_parameter_ptrs)), "shared parameter exists"

        self.optimizer = torch.optim.Adam(
            self.general_parameters,
            lr=hparams.TRAIN.LEARNING_RATE,
            weight_decay=hparams.TRAIN.WEIGHT_DECAY)

        self.offset_optimizer = torch.optim.Adam(self.offset_parameters, lr=hparams.TRAIN.OFFSET.LR)

    def init_dataloader(self):
        self.worker_init_fn = nprand_init
        super().init_dataloader()

    def init_mscoco(self):
        self.coco = COCO("data/mscoco/person_keypoints_train2017.json")
        self.train_dataset = TransformedData(datasets.COCOSinglePose(
            "data/mscoco/images2017",
            self.coco,
            "data/mscoco/sp_split_2017.pth",
            "data/mscoco/" + hparams.DATASET["COCO"].MEAN_STD_FILE,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            ext_border=hparams.DATASET["COCO"].EXT_BORDER,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.IND_TRANSFORMER.KEYPOINT_SIGMA,
            scale_factor=hparams.DATASET["COCO"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["COCO"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["COCO"].TRANSLATION_FACTOR,
            half_body_num_joints=hparams.DATASET.COCO.HALF_BODY_NUM_JOINTS,
            half_body_prob=hparams.DATASET.COCO.HALF_BODY_PROB,
            preserve_transform_data=True))

        self.val_dataset = TransformedData(datasets.COCOSinglePose(
            "data/mscoco/images2017",
            self.coco,
            "data/mscoco/sp_split_2017.pth",
            "data/mscoco/" + hparams.DATASET["COCO"].MEAN_STD_FILE,
            False,
            img_res=hparams.MODEL.INP_SHAPE,
            ext_border=hparams.DATASET["COCO"].EXT_BORDER,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.IND_TRANSFORMER.KEYPOINT_SIGMA,
            scale_factor=hparams.DATASET["COCO"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["COCO"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["COCO"].TRANSLATION_FACTOR,
            preserve_transform_data=True))
        self.train_collate_fn = datasets.COCOSinglePose.collate_function
        self.valid_collate_fn = datasets.COCOSinglePose.collate_function

    def init_mpii(self):
        self.train_dataset = TransformedData(datasets.MPII("data/mpii/images",
            "data/mpii/mpii_human_pose.json",
            "data/mpii/split_sig.pth",
            "data/mpii/mean_std.pth",
            True,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.IND_TRANSFORMER.KEYPOINT_SIGMA,
            scale_factor=hparams.DATASET["MPII"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["MPII"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["MPII"].TRANSLATION_FACTOR,
            preserve_transform_data=True))

        self.val_dataset = TransformedData(datasets.MPII("data/mpii/images",
            "data/mpii/mpii_human_pose.json",
            "data/mpii/split_sig.pth",
            "data/mpii/mean_std.pth",
            False,
            True,
            img_res=hparams.MODEL.INP_SHAPE,
            kpmap_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.IND_TRANSFORMER.KEYPOINT_SIGMA,
            scale_factor=hparams.DATASET["MPII"].SCALE_FACTOR,
            rot_factor=hparams.DATASET["MPII"].ROTATE_FACTOR,
            trans_factor=hparams.DATASET["MPII"].TRANSLATION_FACTOR,
            preserve_transform_data=True))
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
            self.offset_optimizer.load_state_dict(checkpoint["offset_optimizer"])
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_full, epoch, extra=None):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict(),
        }
        if extra:
            checkpoint_dict.update(extra)
        save_checkpoint(checkpoint_dict, checkpoint_full=checkpoint_full, force_replace=True)

    def epoch_start(self, epoch, step, evaluate_only):
        if not evaluate_only:
            cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams.TRAIN.LEARNING_RATE, hparams.TRAIN.SCHEDULE, hparams.TRAIN.LR_GAMMA)
            log_i("Set learning rate to {:.5f}".format(cur_lr))

    def _save_offsets(self, step):
        offset_disabled = True
        for dm in globalvars.displace_mods:
            if dm.learnable_offset:
                offset_disabled = False
        if not offset_disabled:
            torch.save([(dm.get_all_offsets(detach=True) * dm.offset_scale).cpu() for dm in globalvars.displace_mods], os.path.join(globalvars.main_context.checkpoint_dir, "offset_{}.pth".format(step)))
            
    def epoch_end(self, epoch, step, evaluate_only):
        if not evaluate_only:
            self._save_offsets(step)

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        self.optimizer.zero_grad()
        self.offset_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.offset_optimizer.step()

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, progress: dict, train: bool) -> torch.Tensor:
        img = batch["img"]
        img_trans = batch["img_trans"]
        batch_size = img.size(0)

        globalvars.progress = progress

        reg_angle_cos, reg_angle_sin, reg_scale = self.model(torch.cat([img, img_trans], dim=0))
        if config.vis:
            globalvars.img = self.train_dataset.dataset.restore_image(img)
            globalvars.img_trans = self.train_dataset.dataset.restore_image(img_trans)

        angle_loss = 0
        scale_loss = 0
        var_loss = 0

        angle_avg = 0
        scale_avg = 0
        var_avg = 0

        scale = batch["scale"].to(reg_angle_cos, non_blocking=True)
        rotate = batch["rotate"].to(reg_angle_cos, non_blocking=True)
        translation = batch["translation"].to(reg_angle_cos, non_blocking=True)
        mask = batch["mask"].to(reg_angle_cos, non_blocking=True)
        mask_trans = batch["mask_trans"].to(reg_angle_cos, non_blocking=True)

        for scale_factor in hparams.MODEL.IND_TRANSFORMER.MULTI_SCALE:
            if scale_factor == 1:
                reg_angle_cos_cur = reg_angle_cos
                reg_angle_sin_cur = reg_angle_sin
                reg_scale_cur = reg_scale
                mask_cur = mask
                mask_trans_cur = mask_trans
            else:
                interp_kwargs = dict(
                    scale_factor=scale_factor,
                    mode="area" if scale_factor < 1 else "bilinear",
                    align_corners=None if scale_factor < 1 else True)
                reg_angle_cos_cur = F.interpolate(reg_angle_cos, **interp_kwargs)
                reg_angle_sin_cur = F.interpolate(reg_angle_sin, **interp_kwargs)
                reg_scale_cur = F.interpolate(reg_scale, **interp_kwargs)
                mask_cur = F.interpolate(mask, **interp_kwargs)
                mask_trans_cur = F.interpolate(mask_trans, **interp_kwargs)
            angle_loss_cur, scale_loss_cur, var_loss_cur, angle_avg_cur, scale_avg_cur, var_avg_cur = self.loss(
                (reg_angle_cos_cur[:batch_size], reg_angle_sin_cur[:batch_size], reg_scale_cur[:batch_size]),
                (reg_angle_cos_cur[batch_size:], reg_angle_sin_cur[batch_size:], reg_scale_cur[batch_size:]),
                scale,
                rotate,
                translation,
                mask_cur,
                mask_trans_cur
            )
            angle_loss = angle_loss + angle_loss_cur
            scale_loss = scale_loss + scale_loss_cur
            var_loss = var_loss + var_loss_cur

            angle_avg = angle_avg + angle_avg_cur
            scale_avg = scale_avg + scale_avg_cur
            var_avg = var_avg + var_avg_cur

        angle_avg = angle_avg / len(hparams.MODEL.IND_TRANSFORMER.MULTI_SCALE)
        scale_avg = scale_avg / len(hparams.MODEL.IND_TRANSFORMER.MULTI_SCALE)
        var_avg = var_avg / len(hparams.MODEL.IND_TRANSFORMER.MULTI_SCALE)

        if config.vis:
            globalvars.img = None
            globalvars.img_trans = None

        loss = angle_loss * float(hparams.MODEL.IND_TRANSFORMER.LOSS_ANGLE_COF) + \
            scale_loss * float(hparams.MODEL.IND_TRANSFORMER.LOSS_SCALE_COF) + \
            var_loss * float(hparams.MODEL.IND_TRANSFORMER.LOSS_VAR_COF)

        epoch_ctx.set_iter_data("loss", loss)

        epoch_ctx.set_iter_data("avg_angle", angle_avg)
        epoch_ctx.set_iter_data("avg_scale", scale_avg)
        epoch_ctx.set_iter_data("avg_var", var_avg)

        return loss

    def summarize_iter(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")

        loss_val = epoch_ctx.iter_data["loss"].item()
        avg_angle = epoch_ctx.iter_data["avg_angle"].item()
        avg_scale = epoch_ctx.iter_data["avg_scale"].item()
        avg_var = epoch_ctx.iter_data["avg_var"].item()
        if train and tb_writer is not None:
            tb_writer.add_scalar("loss/transformer", loss_val, progress["step"])
            tb_writer.add_scalar("loss/avg_angle", avg_angle, progress["step"])
            tb_writer.add_scalar("loss/avg_scale", avg_scale, progress["step"])
            tb_writer.add_scalar("loss/avg_var", avg_var, progress["step"])

        epoch_ctx.add_scalar("loss", loss_val)
        epoch_ctx.add_scalar("avg_angle", avg_angle)
        epoch_ctx.add_scalar("avg_scale", avg_scale)
        epoch_ctx.add_scalar("avg_var", avg_var)

        if train:
            if (progress["step"] + 1) % hparams.LOG.OFFSET_SAVE_INTERVAL == 0:
                self._save_offsets(progress["step"] + 1)

    def summarize_epoch(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")
        if tb_writer and not train:
            tb_writer.add_scalar("loss/transformer_valid", epoch_ctx.scalar["loss"].avg, progress["step"])
            epoch_ctx.add_scalar("avg_angle", epoch_ctx.scalar["avg_angle"].avg)
            epoch_ctx.add_scalar("avg_scale", epoch_ctx.scalar["avg_scale"].avg)
            epoch_ctx.add_scalar("avg_var", epoch_ctx.scalar["avg_var"].avg)

class TransformedData(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        EPS = np.finfo(np.float32).eps.item()
        data = self.dataset[index]
        img = data["img"]

        center = data["center"]
        img_bgr = data["img_bgr"]
        img_res = data["img_res"]
        img_scale = data["img_scale"]
        img_rotate = data["img_rotate"]
        keypoint_map = data["keypoint_map"]

        scale_std = hparams.TRAIN.IND_TRANSFORMER.SCALE_STD
        rotate_std = float(hparams.TRAIN.IND_TRANSFORMER.ROTATE_STD) / 180 * np.pi
        translation_std = hparams.TRAIN.IND_TRANSFORMER.TRANSLATION_STD

        def get_transformed_data(scale_aug, rotate_aug, translation_aug, mask_extra=None):
            nonlocal self, img, img_bgr, img_res, img_scale, img_rotate, center
            mat_aug = get_transform((0.5 - translation_aug.numpy()) * np.array(img_res), None, (img_res[0], img_res[1]), rot=rotate_aug.item() / np.pi * 180, scale=scale_aug.item())
            img_trans, _, _ = self.dataset.get_transformed_image(img_bgr, img_res, center=center, rotate=img_rotate, scale=img_scale, mat=mat_aug)
            img_trans = torch.from_numpy(img_trans)

            mask = torch.ones(1, img.size(-2) // FACTOR, img.size(-1) // FACTOR, dtype=torch.float)
            if mask_extra is not None:
                mask = mask * mask_extra
            mask_trans = transform_maps(mask[None], scale_aug[None], rotate_aug[None], translation_factor=translation_aug[None])[0]

            return img_trans, mask_trans

        # # truncate at value [max(0, 1-2*std), min(2, 1+2*std)]
        # scale_aug = torch.tensor(truncnorm.rvs(max(-1/(scale_std+EPS), -2), min(1/(scale_std+EPS), 2), loc=1, scale=scale_std)).float()
        # # truncate at value [-3*std, 3*std]
        # rotate_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=rotate_std)).float()
        # translation_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=translation_std, size=2)).float()
        scale_aug = torch.tensor(0., dtype=torch.float).uniform_(max(0, 1-2*scale_std), min(2, 1+2*scale_std))
        rotate_aug = torch.tensor(0., dtype=torch.float).uniform_(-2*rotate_std, 2*rotate_std)
        translation_aug = torch.zeros(2, dtype=torch.float).uniform_(-2*translation_std, 2*translation_std)
        
        mask_extra = keypoint_map.max(dim=0, keepdim=True)[0]
        if hparams.MODEL.IND_TRANSFORMER.LOSS_VAR_COF > 0:
            mask_extra = torch.cat([mask_extra, keypoint_map], dim=0)
        img_trans, mask_trans = get_transformed_data(scale_aug, rotate_aug, translation_aug, mask_extra=mask_extra)

        if config.vis and False:
            import matplotlib.pyplot as plt
            print("scale_aug={}, rotate_aug={}, tranlation_aug={}".format(scale_aug.item(), rotate_aug.item(), translation_aug.tolist()))
            plt.figure()
            plt.imshow(self.dataset.restore_image(img.numpy()))
            plt.figure()
            plt.imshow(self.dataset.restore_image(img_trans.numpy()))
            plt.figure()
            plt.imshow(mask_trans.numpy()[0], vmin=0, vmax=1)
            plt.show()

        return dict(
            img=img,
            img_trans=img_trans,
            mask=mask_extra,
            mask_trans=mask_trans,
            scale=scale_aug,
            rotate=rotate_aug,
            translation=translation_aug
        )

class TransformerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ori, trans, scale, rotate, translation, mask, mask_trans):
        EPS = np.finfo(np.float32).eps.item()

        cos_ori, sin_ori, scale_ori = ori
        cos_trans, sin_trans, scale_trans = trans

        assert scale.dim() == 1 and rotate.dim() == 1 and translation.dim() == 2
        cos_ori_trans_tmp, sin_ori_trans_tmp, scale_ori_trans_tmp = tuple(map(lambda x: transform_maps(x, scale, rotate, translation_factor=translation), ori))

        rotate_sin = torch.sin(rotate)[:, None, None, None]
        rotate_cos = torch.cos(rotate)[:, None, None, None]
        cos_ori_trans = cos_ori_trans_tmp * rotate_cos - sin_ori_trans_tmp * rotate_sin
        sin_ori_trans = cos_ori_trans_tmp * rotate_sin + sin_ori_trans_tmp * rotate_cos
        scale_ori_trans = scale_ori_trans_tmp * scale[:, None, None, None]
        del cos_ori_trans_tmp, sin_ori_trans_tmp, scale_ori_trans_tmp

        norm_ori_trans = (cos_ori_trans.pow(2) + sin_ori_trans.pow(2)).detach().sqrt() + EPS
        cos_ori_trans = cos_ori_trans / norm_ori_trans
        sin_ori_trans = sin_ori_trans / norm_ori_trans

        mask_sep = None
        mask_sep_trans = None
        if mask_trans.size(1) > 1:
            mask_sep = mask[:, 1:]
            mask_sep_trans = mask_trans[:, 1:]
            mask_max_trans = mask_trans[:, [0]]

        dis_cossim = lambda cos_a, sin_a, cos_b, sin_b: (1 - cos_a * cos_b - sin_a * sin_b)
        dis_angle = lambda cos_a, sin_a, cos_b, sin_b: (cos_a * cos_b + sin_a * sin_b).clamp(-1+EPS, 1-EPS).acos()
        dis_scale = lambda scale_a, scale_b: torch.log(scale_a / (scale_b + EPS) + EPS)

        mask_max_trans_sum = mask_max_trans.sum()
        if mask_max_trans_sum > 0:
            angle_loss = (dis_cossim(cos_ori_trans, sin_ori_trans, cos_trans, sin_trans).pow(2) * mask_max_trans).sum() / mask_max_trans_sum
            scale_loss = (dis_scale(scale_ori_trans, scale_trans).pow(2) * mask_max_trans).sum() / mask_max_trans_sum
            
            angle_avg = ((dis_angle(cos_ori_trans, sin_ori_trans, cos_trans, sin_trans).abs() * mask_max_trans).sum() / mask_max_trans_sum).detach()
            scale_avg = ((dis_scale(scale_ori_trans, scale_trans).abs() * mask_max_trans).sum() / mask_max_trans_sum).detach()
        else:
            angle_loss = torch.tensor(0).to(cos_ori, non_blocking=True)
            scale_loss = torch.tensor(0).to(cos_ori, non_blocking=True)

            angle_avg = torch.tensor(0).to(cos_ori, non_blocking=True)
            scale_avg = torch.tensor(0).to(cos_ori, non_blocking=True)

        var_loss = torch.tensor(0).to(cos_ori, non_blocking=True)
        var_avg = torch.tensor(0).to(cos_ori, non_blocking=True)
        if hparams.MODEL.IND_TRANSFORMER.LOSS_VAR_COF:
            assert mask_sep is not None
            for i in range(mask_sep.size(1)):
                mask_i = mask_sep[:, [i]]
                mask_i_spa_sum = mask_i.sum(dim=-1).sum(dim=-1)
                valid_sample = (mask_i_spa_sum[:, 0] > 0).nonzero()[:, 0]
                if len(valid_sample) > 0:
                    # calc avg angle for each keypoint
                    cos_i_avg = (cos_ori * mask_i).sum(dim=-1).sum(dim=-1) / (mask_i_spa_sum + EPS)
                    sin_i_avg = (sin_ori * mask_i).sum(dim=-1).sum(dim=-1) / (mask_i_spa_sum + EPS)
                    # calc distance to avg angle for each keypoint
                    diff = (cos_ori * cos_i_avg[:, :, None, None] + sin_ori * sin_i_avg[:, :, None, None]).clamp(-1+EPS, 1-EPS).acos()
                    var_loss = var_loss + (diff.pow(2) * mask_i)[valid_sample].sum()
                    var_avg = var_avg + (diff.abs() * mask_i)[valid_sample].sum().detach()
            mask_sep_sum = mask_sep.sum()
            if mask_sep_sum > 0:
                var_loss = var_loss / mask_sep_sum
                var_avg = var_avg / mask_sep_sum

        if config.vis: # globalvars.progress["step"] > 500 or 
            import matplotlib.pyplot as plt
            from matplotlib.colors import hsv_to_rgb
            img_h_ori = torch.atan2(sin_ori, cos_ori) / np.pi / 2 + 0.5
            img_h_ori_trans = torch.atan2(sin_ori_trans, cos_ori_trans) / np.pi / 2 + 0.5
            img_h_trans = torch.atan2(sin_trans, cos_trans) / np.pi / 2 + 0.5
            
            cos_sim = ((cos_ori_trans * cos_trans + sin_ori_trans * sin_trans - 0.5) / 0.5 * mask_max_trans).detach().clamp(0, 1)
            scale_sim = ((0.3 - torch.log(scale_ori_trans / (scale_trans + EPS) + EPS).abs()) / 0.3 * mask_max_trans).detach().clamp(0, 1)
            for i in range(cos_trans.size(0)):
                fig, axes = plt.subplots(2, 5, figsize=(30, 16))
                axes[0, 0].imshow(globalvars.img[i])
                axes[1, 0].imshow(globalvars.img_trans[i])
                axes[0, 1].imshow(hsv_to_rgb(np.stack((img_h_ori[i, 0].detach().cpu().numpy(), np.ones(img_h_ori.shape[-2:]), np.ones(img_h_ori.shape[-2:])), axis=-1)))
                axes[1, 1].imshow(scale_ori[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                axes[0, 2].imshow(hsv_to_rgb(np.stack((img_h_ori_trans[i, 0].detach().cpu().numpy(), np.ones(img_h_ori_trans.shape[-2:]), np.ones(img_h_ori_trans.shape[-2:])), axis=-1)))
                axes[1, 2].imshow(hsv_to_rgb(np.stack((img_h_trans[i, 0].detach().cpu().numpy(), np.ones(img_h_trans.shape[-2:]), np.ones(img_h_trans.shape[-2:])), axis=-1)))
                axes[0, 3].imshow(scale_ori_trans[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                axes[1, 3].imshow(scale_trans[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                axes[0, 4].imshow(cos_sim[i, 0].detach().cpu().numpy(), vmin=0, vmax=1)
                axes[1, 4].imshow(scale_sim[i, 0].detach().cpu().numpy(), vmin=0, vmax=1)
                plt.show()

        if config.check and (torch.isnan(angle_loss).any() or torch.isnan(scale_loss).any()):
            raise RuntimeError("NaN")

        return angle_loss, scale_loss, var_loss, angle_avg, scale_avg, var_avg

        