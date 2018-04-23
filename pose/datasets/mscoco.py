from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import math
import cv2
import copy

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
from pycocotools.coco import COCO
import pycocotools

FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

PART_LABELS = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']

NUM_PARTS = 17

class COCOPose(data.Dataset):
    def __init__(self, img_folder, anno, split_file, meanstd_file,
                 train, single_person,
                 img_res=[256], return_img_transform=False,
                 kpmap_res=64, locmap_res=0, mask_res=0,
                 kpmap_select=None, kpmap_sigma=1, locmap_min_sigma=0.5,
                 keypoint_res=0, locate_res=0,
                 scale_factor=0.25, rot_factor=30, person_random_selection=False):
        assert not single_person
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.single_person = single_person
        self.img_res = img_res
        self.return_img_transform = return_img_transform
        self.kpmap_res = kpmap_res
        self.locmap_res = locmap_res
        self.mask_res = mask_res
        self.kpmap_select = kpmap_select
        self.kpmap_sigma = kpmap_sigma
        self.locmap_min_sigma = locmap_min_sigma
        self.keypoint_res = keypoint_res
        self.locate_res = locate_res
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.person_random_selection = person_random_selection
        if self.person_random_selection:
            assert not self.single_person

        self.heatmap_gen = HeatmapGenerator(self.kpmap_res, self.kpmap_sigma)

        # create train/val split
        if isinstance(anno, COCO):
            self.coco = anno
        else:
            self.coco = COCO(anno)

        self.train, self.valid = self._split(split_file)
        self.mean, self.std = self._compute_mean(meanstd_file)

    def _split(self, split_file):
        if split_file is not None and os.path.isfile(split_file):
            split = torch.load(split_file)
            train = split["train"]
            valid = split["valid"]
        else:
            raise ValueError("%s not found" % (split_file,))

        return train, valid

    def _load_image(self, img_index, bgr=True):
        img_info = self.coco.loadImgs(img_index)[0]
        path = img_info['file_name']
        img_file = os.path.join(self.img_folder, path)
        img_bgr = cv2.imread(img_file)
        return img_bgr if bgr else img_bgr[..., ::-1]

    def _compute_mean(self, meanstd_file):
        if meanstd_file is None:
            print("Warning: not using mean_std")
            return np.array([0.]*3), np.array([1.]*3)

        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            print("Creating mean_std...")
            mean = np.zeros(3)
            std = np.zeros(3)
            for img_index in self.train:
                img = self._load_image(img_index, bgr=False)
                mean += img.reshape((-1, 3)).mean(0)
                std += img.reshape((-1, 3)).std(0)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % \
                    (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % \
                    (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
        assert type(meanstd['mean']) is np.ndarray
        return meanstd['mean'].astype(np.float32)/255, meanstd['std'].astype(np.float32)/255

    def _draw_label(self, points, target_map, point_type="person", sigma=None, out_res=None):
        # Generate ground truth
        if point_type == "person":
            assert sigma is None
            for ijoint in range(points.shape[0]):
                if points[ijoint, 2] > 0:
                    self.heatmap_gen(points[ijoint, :2], ijoint, target_map)
        elif point_type == "point":
            assert not np.isclose(sigma, 0)
            self.heatmap_gen(points, 0, target_map, sigma=sigma, out_res=out_res, normalize_factor=0)
        else:
            raise RuntimeError("Wrong point_type")

    def _compute_locate_mean_std(self, keypoints):
        locate_mean = list()
        locate_std = list()
        locate_in_kp = list()
        for iperson, points in enumerate(keypoints):
            labeled_mask = (points[:, 2] > 0)
            labeled_points = points[labeled_mask.nonzero()]
            if labeled_points.shape[0] > 0:
                labeled_points = labeled_points[:, :2].astype(np.float32)
                mean = labeled_points.mean(axis=0)
                if labeled_points.shape[0] > 1:
                    std = np.sqrt(((labeled_points - mean) ** 2).sum(axis=1)).mean()
                    if std < self.locmap_min_sigma:
                        std = self.locmap_min_sigma
                else:
                    std = None
                inside_mask = ((labeled_points[:, 0] >= 0) & (labeled_points[:, 0] < self.kpmap_res) &
                               (labeled_points[:, 1] >= 0) & (labeled_points[:, 1] < self.kpmap_res))
                labeled_points = labeled_points[inside_mask.nonzero()]
                if labeled_points.shape[0] > 0:
                    mean = labeled_points.mean(axis=0)
                    locate_mean.append(mean)
                    locate_std.append(std)
                    locate_in_kp.append(iperson)
            else:
                raise AssertionError("A person has no label")
        if len(locate_mean) > 0:
            locate_mean = np.stack(locate_mean, axis=0)
            locate_in_kp = np.array(locate_in_kp, dtype=np.int64)
        else:
            locate_mean = np.zeros((0, 2), dtype=np.float32)
            locate_in_kp = np.zeros((0,), dtype=np.int64)
        return locate_mean, locate_std, locate_in_kp

    def __getitem__(self, index):
        sf = float(self.scale_factor)
        rf = float(self.rot_factor)

        if self.is_train:
            img_index = self.train[index]
        else:
            img_index = self.valid[index]

        # =====
        # Image

        img_bgr = self._load_image(img_index, bgr=True)

        img_size = np.array(list(img_bgr.shape[:2][::-1]), dtype=np.float32) # W, H

        center = img_size / 2
        scale = float(img_size.max()) / 200
        rotate = 0
        flip_status = False

        # Image augmentation
        if self.is_train:
            scale = scale * ((np.random.randn(1) * sf) + 1).clip(1-sf, 1+sf)[0]
            rotate = (np.random.randn(1) * rf).clip(-2*rf, 2*rf)[0] \
                    if np.random.rand() <= 0.6 else 0

            center += np.random.randint(-40 * scale, 40 * scale, size=2)

            # Color distort
            img_bgr = (img_bgr.astype(np.float32) * (np.random.rand(3) * 0.4 + 0.8)).round().clip(0, 255).astype(np.uint8)

            # Flip
            if np.random.rand() <= 0.5:
                flip_status = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        # Prepare image and groundtruth map
        img_list = list()
        if self.return_img_transform:
            img_transform_list = list()
        for cur_res in (self.img_res if isinstance(self.img_res, list) else [self.img_res]):
            img_transform_mat = get_transform(center, scale, [cur_res, cur_res], rot=rotate)
            if self.return_img_transform:
                img_transform_list.append(img_transform_mat)
            img_bgr_warp = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(cur_res, cur_res), flags=cv2.INTER_LINEAR)
            img = img_bgr_warp[..., ::-1].astype(np.float32) / 255
            # Color normalize
            img -= self.mean
            img = img.transpose(2, 0, 1)
            img_list.append(img)

        # =====
        # Keypoint Map

        ann_ids = self.coco.getAnnIds(imgIds=img_index)

        anns = self.coco.loadAnns(ann_ids)

        keypoints = np.array([ann['keypoints'] for ann in anns \
                              if ann["num_keypoints"] > 0], dtype=np.float32) \
                .reshape((-1, NUM_PARTS, 3))

        if self.person_random_selection:
            person_indices = np.arange(keypoints.shape[0])
            np.random.shuffle(person_indices)
            keypoints = keypoints[person_indices[:np.random.randint(low=1, high=person_indices.shape[0]+1)]]

        if flip_status:
            keypoints = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))

        # keypoints: #person * #joints * 3
        keypoints_tf = transform(keypoints.reshape((-1, 3))[...,:2], center, scale, [self.kpmap_res, self.kpmap_res], rot=rotate)
        keypoints_tf = np.c_[keypoints_tf, keypoints.reshape((-1, 3))[:, 2]].reshape(keypoints.shape)

        keypoints_tf_ret = keypoints_tf.copy()
        if self.keypoint_res > 0 and self.keypoint_res != self.kpmap_res:
            assert keypoints_tf_ret.dtype == np.float32
            keypoints_tf_ret[:, :, :2] = keypoints_tf_ret[:, :, :2] * (float(self.keypoint_res) / self.kpmap_res)

        locate_mean, locate_std, locate_in_kp = self._compute_locate_mean_std(keypoints_tf)
        locate_mean_ret, locate_std_ret = copy.deepcopy(locate_mean), copy.deepcopy(locate_std)
        if self.locate_res > 0 and self.locate_res != self.kpmap_res:
            assert locate_mean_ret.dtype == np.float32
            locate_mean_ret = locate_mean_ret * (float(self.locate_res) / self.kpmap_res)
            locate_std_ret = [(lsr * (float(self.locate_res) / self.kpmap_res)) if lsr is not None else None for lsr in locate_std_ret]

        if self.kpmap_select is None:
            draw_parts = []
        elif self.kpmap_select == "all":
            assert self.kpmap_res > 0
            draw_parts = list(range(NUM_PARTS))
        elif isinstance(self.kpmap_select, list):
            assert self.kpmap_res > 0
            draw_parts = self.kpmap_select
        else:
            raise ValueError("Wrong kpmap_select")

        kp_map = None
        if len(draw_parts) > 0:
            kp_map = np.zeros((len(draw_parts), self.kpmap_res, self.kpmap_res), dtype=np.float32)
            for iperson, points in enumerate(keypoints_tf):
                self._draw_label(points[draw_parts], kp_map)

        loc_map = None
        if self.locmap_res > 0:
            if self.locmap_res != self.kpmap_res:
                locate_mean = locate_mean * (float(self.locmap_res) / self.kpmap_res)
                locate_std = [(lsr * (float(self.locmap_res) / self.kpmap_res)) if lsr is not None else None for lsr in locate_std]
            loc_map = np.zeros((1, self.locmap_res, self.locmap_res), dtype=np.float32)
            for lmean, lstd in zip(locate_mean, locate_std):
                self._draw_label(lmean, loc_map, point_type="point", sigma=np.sqrt(float(lstd)/3) if lstd is not None else 1, out_res=self.locmap_res)
        # =====
        # Mask: used to mask off crowds

        # init mask with image size
        mask_crowd = None
        if self.mask_res > 0:
            mask_crowd = np.zeros(img_size[::-1].astype(int))
            for ann in anns:
                if ann['iscrowd']:
                    mask_crowd += self.coco.annToMask(ann)

            mask_crowd = (mask_crowd > 0.5).astype(np.float32)

            if flip_status:
                mask_crowd = fliplr_chwimg(mask_crowd)

            mask_transform_mat = get_transform(center, scale, [self.mask_res, self.mask_res], rot=rotate)
            mask_crowd = cv2.warpAffine((mask_crowd*255).clip(0, 255).astype(np.uint8), mask_transform_mat[:2], dsize=(self.mask_res, self.mask_res), flags=cv2.INTER_LINEAR).astype(np.float32) / 255

            mask_crowd = (mask_crowd > 0.5)
            mask_noncrowd = (~mask_crowd).astype(np.uint8)

        result = {
            "index": index,
            "img_index": img_index,
            "img": [torch.from_numpy(img) for img in img_list] if isinstance(self.img_res, list) else torch.from_numpy(img_list[0]),
            "center": torch.from_numpy(center),
            "scale": scale,
            "keypoint_ori": torch.from_numpy(keypoints) if keypoints.shape[0] > 0 else None,
            "keypoint": torch.from_numpy(keypoints_tf_ret) if keypoints.shape[0] > 0 else None,
        }

        if self.return_img_transform:
            result["img_transform"] = [torch.from_numpy(mat) for mat in img_transform_list] if isinstance(self.img_res, list) else torch.from_numpy(img_transform_list[0])
            result["img_flipped"] = flip_status

        if mask_noncrowd is not None:
            result["mask"] = torch.from_numpy(mask_noncrowd)

        if kp_map is not None:
            result["keypoint_map"] = torch.from_numpy(kp_map)

        if loc_map is not None:
            result["locate_map"] = torch.from_numpy(loc_map)

        if self.locate_res > 0:
            result["locate"] = torch.from_numpy(locate_mean_ret) if locate_mean_ret.shape[0] > 0 else None
            result["locate_std"] = locate_std_ret if len(locate_std_ret) > 0 else None
            result["locate_in_kp"] = torch.from_numpy(locate_in_kp).long() if locate_in_kp.shape[0] > 0 else None

        return result

    @classmethod
    def collate_function(cls, batch):
        NON_COLLATE_KEYS = ["keypoint_ori", "keypoint", "locate", "locate_std", "locate_in_kp", "img_transform", "img_flipped"]
        collate_fn = data.dataloader.default_collate
        all_keys = batch[0].keys()
        result = dict()
        for k in all_keys:
            if k in NON_COLLATE_KEYS:
                result[k] = [sample[k] for sample in batch]
            else:
                result[k] = collate_fn([sample[k] for sample in batch])
        return result

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
