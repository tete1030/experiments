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
from lib.utils.augtrans import transform_maps
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
            kpmap_res=None,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            kpmap_res=None,
            keypoint_res=hparams.MODEL.OUT_SHAPE,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            kpmap_res=None,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
            kpmap_res=None,
            kpmap_sigma=hparams.MODEL.GAUSSIAN_KERNELS,
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
        angle_loss, scale_loss = self.loss(
            (reg_angle_cos[:batch_size], reg_angle_sin[:batch_size], reg_scale[:batch_size]),
            (reg_angle_cos[batch_size:], reg_angle_sin[batch_size:], reg_scale[batch_size:]),
            batch["scale"].to(reg_angle_cos, non_blocking=True),
            batch["rotate"].to(reg_angle_cos, non_blocking=True),
            batch["translation"].to(reg_angle_cos, non_blocking=True),
            batch["mask_trans"].to(reg_angle_cos, non_blocking=True)
        )

        loss = angle_loss * float(hparams.MODEL.IND_TRANSFORMER.LOSS_ANGLE_COF) + \
            scale_loss * float(hparams.MODEL.IND_TRANSFORMER.LOSS_SCALE_COF)

        epoch_ctx.set_iter_data("loss_angle", angle_loss)
        epoch_ctx.set_iter_data("loss_scale", scale_loss)
        epoch_ctx.set_iter_data("loss", loss)

        return loss

    def summarize_iter(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")

        loss_val = epoch_ctx.iter_data["loss"].item()
        if train and tb_writer is not None:
            tb_writer.add_scalar("loss/transformer", loss_val, progress["step"])

        epoch_ctx.add_scalar("loss", loss_val)
        epoch_ctx.add_scalar("loss_angle", epoch_ctx.iter_data["loss_angle"].item())
        epoch_ctx.add_scalar("loss_scale", epoch_ctx.iter_data["loss_scale"].item())

        if train:
            if (progress["step"] + 1) % hparams.LOG.OFFSET_SAVE_INTERVAL == 0:
                self._save_offsets(progress["step"] + 1)

    def summarize_epoch(self, epoch_ctx:EpochContext, progress:dict, train:bool):
        tb_writer = globalvars.main_context.get("tb_writer")
        if tb_writer and not train:
            tb_writer.add_scalar("loss/transformer_valid", epoch_ctx.scalar["loss"].avg, progress["step"])

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

        scale_std = hparams.TRAIN.IND_TRANSFORMER.SCALE_STD
        rotate_std = float(hparams.TRAIN.IND_TRANSFORMER.ROTATE_STD) / 180 * np.pi
        translation_std = hparams.TRAIN.IND_TRANSFORMER.TRANSLATION_STD

        # # truncate at value [max(0, 1-2*std), min(2, 1+2*std)]
        # scale_aug = torch.tensor(truncnorm.rvs(max(-1/(scale_std+EPS), -2), min(1/(scale_std+EPS), 2), loc=1, scale=scale_std)).float()
        # # truncate at value [-3*std, 3*std]
        # rotate_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=rotate_std)).float()
        # translation_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=translation_std, size=2)).float()
        scale_aug = torch.tensor(0., dtype=torch.float).uniform_(max(0, 1-2*scale_std), min(2, 1+2*scale_std))
        rotate_aug = torch.tensor(0., dtype=torch.float).uniform_(-2*rotate_std, 2*rotate_std)
        translation_aug = torch.zeros(2, dtype=torch.float).uniform_(-2*translation_std, 2*translation_std)

        mat_aug = get_transform((0.5 - translation_aug.numpy()) * np.array(img_res), None, (img_res[0], img_res[1]), rot=rotate_aug.item() / np.pi * 180, scale=scale_aug.item())
        img_trans, _, _ = self.dataset.get_transformed_image(img_bgr, img_res, center=center, rotate=img_rotate, scale=img_scale, mat=mat_aug)
        img_trans = torch.from_numpy(img_trans)

        mask = torch.ones(1, img.size(-2) // FACTOR, img.size(-1) // FACTOR, dtype=torch.float)
        mask_trans = ((transform_maps(mask[None], scale_aug[None], rotate_aug[None], translation_factor=translation_aug[None]))[0] >= 0.99).float()

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
            mask_trans=mask_trans,
            scale=scale_aug,
            rotate=rotate_aug,
            translation=translation_aug
        )

class TransformerLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ori, trans, scale, rotate, translation, mask_trans):
        EPS = np.finfo(np.float32).eps.item()

        cos_ori, sin_ori, scale_ori = ori
        cos_trans, sin_trans, scale_trans = trans

        assert scale.dim() == 1 and rotate.dim() == 1 and translation.dim() == 2
        cos_ori_trans, sin_ori_trans, scale_ori_trans = tuple(map(lambda x: transform_maps(x, scale, rotate, translation_factor=translation), ori))

        rotate_sin = torch.sin(rotate)[:, None, None, None]
        rotate_cos = torch.cos(rotate)[:, None, None, None]
        cos_ori_trans = cos_ori_trans * rotate_cos - sin_ori_trans * rotate_sin
        sin_ori_trans = cos_ori_trans * rotate_sin + sin_ori_trans * rotate_cos
        scale_ori_trans = scale_ori_trans * scale[:, None, None, None]

        if config.vis and False: # globalvars.progress["step"] > 500 or 
            import matplotlib.pyplot as plt
            from matplotlib.colors import hsv_to_rgb
            img_h_ori = torch.atan2(sin_ori, cos_ori) / np.pi / 2 + 0.5
            img_h_ori_trans = torch.atan2(sin_ori_trans, cos_ori_trans) / np.pi / 2 + 0.5
            img_h_trans = torch.atan2(sin_trans, cos_trans) / np.pi / 2 + 0.5
            for i in range(cos_trans.size(0)):
                plt.figure()
                plt.imshow(hsv_to_rgb(np.stack((img_h_ori[i, 0].detach().cpu().numpy(), np.ones(img_h_ori.shape[-2:]), np.ones(img_h_ori.shape[-2:])), axis=-1)))
                plt.figure()
                plt.imshow(hsv_to_rgb(np.stack((img_h_ori_trans[i, 0].detach().cpu().numpy(), np.ones(img_h_ori_trans.shape[-2:]), np.ones(img_h_ori_trans.shape[-2:])), axis=-1)))
                plt.figure()
                plt.imshow(hsv_to_rgb(np.stack((img_h_trans[i, 0].detach().cpu().numpy(), np.ones(img_h_trans.shape[-2:]), np.ones(img_h_trans.shape[-2:])), axis=-1)))
                plt.figure()
                plt.imshow(scale_ori[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                plt.figure()
                plt.imshow(scale_ori_trans[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                plt.figure()
                plt.imshow(scale_trans[i, 0].detach().cpu().numpy(), vmin=0, vmax=2)
                plt.show()

        angle_loss = ((1 - cos_ori_trans * cos_trans - sin_ori_trans * sin_trans) * mask_trans).sum() / mask_trans.sum()
        scale_loss = (torch.log(scale_ori_trans / (scale_trans + EPS) + EPS).abs() * mask_trans).sum() / mask_trans.sum()

        if config.check and (torch.isnan(angle_loss).any() or torch.isnan(scale_loss).any()):
            raise RuntimeError("NaN")

        return angle_loss, scale_loss

        