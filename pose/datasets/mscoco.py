from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import math
import cv2

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
    def __init__(self, img_folder, anno_file, split_file, meanstd_file,
                 train, single_person,
                 inp_res=256, out_res=64,
                 label_sigma=1, scale_factor=0.25, rot_factor=30):
        assert single_person == False
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.label_sigma = label_sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.single_person = single_person
        self.heatmap_gen = HeatmapGenerator(self.out_res, self.label_sigma)

        # create train/val split
        self.coco = COCO(anno_file)

        self.train, self.valid = self._split(split_file) 
        self.mean, self.std = self._compute_mean(meanstd_file)

    def _split(self, split_file):
        if split_file is not None and os.path.isfile(split_file):
            split = torch.load(split_file)
            train = split["train"]
            valid = split["valid"]
        else:
            imgids = self.coco.getImgIds()
            train = list()
            valid = list()
            for imgid in imgids:
                ann_ids = self.coco.getAnnIds(imgIds=imgid)
                anns = self.coco.loadAnns(ann_ids)
                ann_count = 0
                for ann in anns:
                    if ann["num_keypoints"] > 0:
                        ann_count += 1
                        break
                if ann_count > 0:
                    train.append(imgid)
            if split_file is not None:
                torch.save({"train": train, "valid": valid}, split_file)
        return train, valid

    def _load_image(self, img_index, bgr=True):
        img_info = self.coco.loadImgs(img_index)[0]
        path = img_info['file_name']
        img_file = os.path.join(self.img_folder, path)
        img_bgr = cv2.imread(img_file)
        return img_bgr if bgr else img_bgr[...,::-1]

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
        return meanstd['mean'], meanstd['std']
 
    def _draw_label(self, points, target_map):
        # Generate ground truth
        for ijoint in range(points.shape[0]):
            if points[ijoint, 2] > 0:
                self.heatmap_gen(points[ijoint, :2], ijoint, target_map)

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor

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
        flip = False

        # Image augmentation
        if self.is_train:
            scale = scale * ((np.random.randn(1) * sf) + 1).clip(1-sf, 1+sf)[0]
            rotate = (np.random.randn(1) * rf).clip(-2*rf, 2*rf)[0] \
                    if np.random.rand() <= 0.6 else 0

            center += np.random.randint(-40 * scale, 40 * scale, size=2)

            # Flip
            if np.random.rand() <= 0.5:
                flip = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        # Prepare image and groundtruth map
        img_transform_mat = get_transform(center, scale, [self.inp_res, self.inp_res], rot=rotate)
        img_bgr = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(self.inp_res, self.inp_res), flags=cv2.INTER_LINEAR)

        img = img_bgr[..., ::-1].astype(np.float32) / 255

        if self.is_train:
            # Color distort
            img = (img * (np.random.rand(3) * 0.4 + 0.8)).clip(0, 1).astype(np.float32) 

        # Color normalize
        img -= self.mean

        img = img.transpose(2, 0, 1)

        # =====
        # Keypoint Map

        ann_ids = self.coco.getAnnIds(imgIds=img_index)

        anns = self.coco.loadAnns(ann_ids)

        keypoints = np.array([ann['keypoints'] for ann in anns \
                              if ann["num_keypoints"] > 0]) \
                .reshape((-1, NUM_PARTS, 3))

        if flip:
            keypoints = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))

        # keypoints: #person * #joints * 3
        keypoints_tf = transform(keypoints.reshape((-1, 3))[...,:2], center, scale, [self.out_res, self.out_res], rot=rotate)
        keypoints_tf = np.c_[keypoints_tf, keypoints.reshape((-1, 3))[:, 2]].reshape(keypoints.shape)
        
        target_map = np.zeros((NUM_PARTS, self.out_res, self.out_res), dtype=np.float32)

        for iperson, points in enumerate(keypoints_tf):
            self._draw_label(points, target_map)

        # =====
        # Mask: used to mask off crowds

        # init mask with image size
        mask_crowd = np.zeros(img_size[::-1].astype(int))
        for ann in anns:
            if ann['iscrowd']:
                mask_crowd += self.coco.annToMask(ann)

        mask_crowd = (mask_crowd > 0.5).astype(np.float32)

        if flip:
            mask_crowd = fliplr_chwimg(mask_crowd)

        mask_transform_mat = get_transform(center, scale, [self.out_res, self.out_res], rot=rotate)
        mask_crowd = cv2.warpAffine((mask_crowd*255).clip(0, 255).astype(np.uint8), mask_transform_mat[:2], dsize=(self.out_res, self.out_res), flags=cv2.INTER_LINEAR).astype(np.float32) / 255
        
        mask_crowd = (mask_crowd > 0.5)
        mask_noncrowd = (~mask_crowd).astype(np.uint8)

        extra = {'index': index, 'center': torch.from_numpy(center), 'scale': scale, 
                 'keypoints': torch.from_numpy(keypoints) if keypoints.shape[0] > 1 else None,
                 'keypoints_tf': torch.from_numpy(keypoints_tf) if keypoints.shape[0] > 1 else None}

        return torch.from_numpy(img), \
               torch.from_numpy(target_map), \
               torch.from_numpy(mask_noncrowd), \
               extra

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
