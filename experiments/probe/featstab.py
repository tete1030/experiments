import threading
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.globals import config, hparams, globalvars
from utils.log import log_w
from lib.utils.transforms import get_transform
from lib.utils.augtrans import transform_maps

def init_feat_stab():
    globalvars.feat_stab_outputs = dict()
    globalvars.feat_stab_lock = threading.Lock()
    globalvars.feat_stab_running = False

def save_feat_stab(var_name, var_obj):
    globalvars.feat_stab_lock.acquire()
    if var_name not in globalvars.feat_stab_outputs:
        globalvars.feat_stab_outputs[var_name] = dict()
    if var_obj.device not in globalvars.feat_stab_outputs[var_name]:
        globalvars.feat_stab_outputs[var_name][var_obj.device] = list()
    globalvars.feat_stab_outputs[var_name][var_obj.device].append(var_obj)
    globalvars.feat_stab_lock.release()

def clear_feat_stab():
    globalvars.feat_stab_lock.acquire()
    globalvars.feat_stab_outputs = dict()
    globalvars.feat_stab_lock.release()

def extract_feat_stab(var_name, device):
    globalvars.feat_stab_lock.acquire()
    if var_name in globalvars.feat_stab_outputs:
        feat_stabs = globalvars.feat_stab_outputs[var_name][device]
    else:
        feat_stabs = None
    globalvars.feat_stab_lock.release()
    return feat_stabs


class TransformedData(Dataset):
    def __init__(self, dataset, scale_std, rotate_std, translation_std, use_keypoint_mask=False):
        super().__init__()
        self.dataset = dataset
        self.dataset.preserve_transform_data = True
        self.scale_std = scale_std
        self.rotate_std = rotate_std
        self.translation_std = translation_std
        self.use_keypoint_mask = use_keypoint_mask
        self._keypoint_map_warning_showed = False

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
        if isinstance(keypoint_map, list):
            keypoint_map = keypoint_map[0]
            if not self._keypoint_map_warning_showed:
                self._keypoint_map_warning_showed = True
                log_w("Only first object of 'keypoint_map' is used.")

        def get_transformed_data(scale_aug, rotate_aug, translation_aug, mask):
            nonlocal self, img, img_bgr, img_res, img_scale, img_rotate, center
            mat_aug = get_transform((0.5 - translation_aug.numpy()) * np.array(img_res), None, (img_res[0], img_res[1]), rot=rotate_aug.item() / np.pi * 180, scale=scale_aug.item())
            img_trans, _, _ = self.dataset.get_transformed_image(img_bgr, img_res, center=center, rotate=img_rotate, scale=img_scale, mat=mat_aug)
            img_trans = torch.from_numpy(img_trans)

            mask_trans = transform_maps(mask[None], scale_aug[None], rotate_aug[None], translation_factor=translation_aug[None])[0]

            return img_trans, mask_trans

        # # truncate at value [max(0, 1-2*std), min(2, 1+2*std)]
        # scale_aug = torch.tensor(truncnorm.rvs(max(-1/(self.scale_std+EPS), -2), min(1/(self.scale_std+EPS), 2), loc=1, scale=self.scale_std)).float()
        # # truncate at value [-3*std, 3*std]
        # rotate_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=self.rotate_std)).float()
        # translation_aug = torch.tensor(truncnorm.rvs(-2, 2, loc=0, scale=self.translation_std, size=2)).float()
        scale_aug = torch.tensor(0., dtype=torch.float).uniform_(max(0, 1-2*self.scale_std), min(2, 1+2*self.scale_std))
        rotate_aug = torch.tensor(0., dtype=torch.float).uniform_(-2*self.rotate_std, 2*self.rotate_std)
        translation_aug = torch.zeros(2, dtype=torch.float).uniform_(-2*self.translation_std, 2*self.translation_std)
        
        if self.use_keypoint_mask:
            mask = keypoint_map.max(dim=0, keepdim=True)[0]
        else:
            mask = torch.ones(1, *keypoint_map.size()[-2:], dtype=torch.float)
        img_trans, mask_trans = get_transformed_data(scale_aug, rotate_aug, translation_aug, mask)

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
            **data,
            td_img_trans=img_trans,
            td_mask=mask,
            td_mask_trans=mask_trans,
            td_scale=scale_aug,
            td_rotate=rotate_aug,
            td_translation=translation_aug
        )

class FeatureStabLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ori_features, new_features, scale, rotate, translation, mask_trans):
        mask_trans_resized = dict()

        loss = 0

        for pair in zip(ori_features, new_features):
            ori = pair[0]
            new = pair[1]

            ori_trans = transform_maps(ori, scale, rotate, translation_factor=translation)
            feat_size = tuple(ori.size()[-2:])
            if feat_size not in mask_trans_resized:
                mask_trans_resized[feat_size] = torch.nn.functional.interpolate(mask_trans, feat_size, mode='area')
            loss = loss + ((ori_trans - new) ** 2 * mask_trans_resized[feat_size]).mean()

        return loss
