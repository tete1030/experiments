#!python3

import os
import json
import cv2
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.utils.data as data

from pose.utils.imutils import HeatmapGenerator
from pose.utils.transforms import get_transform, transform, fliplr_pts

FLIP_INDEX = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

PART_LABELS = ['ank_r', 'kne_r', 'hip_r', 'hip_l', 'kne_l', 'ank_l',
               'pelv', 'thor', 'upnk', 'htop',
               'wri_r', 'elb_r', 'sho_r', 'sho_l', 'elb_l', 'wri_l']

NUM_PARTS = 16

EVAL_INDEX = [0, 1, 2, 3, 4, 5, 10, 11, 14, 15]

class MPII(data.Dataset):
    def __init__(self, img_folder, anno_file, split_file, meanstd_file,
                 is_train, single_person,
                 img_res=(192, 256), kpmap_res=(48, 64),
                 kpmap_sigma=1., scale_factor=0.25, rot_factor=30, trans_factor=0.05):
        self.img_folder = img_folder    # root image folders
        self.is_train = is_train           # training set or test set
        self.img_res = tuple(img_res)
        self.kpmap_res = tuple(kpmap_res)
        self.kpmap_sigma = kpmap_sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.single_person = single_person
        self.heatmap_gen = HeatmapGenerator(out_res=self.kpmap_res)

        # create train/val split
        with open(anno_file) as af:
            self.mpii = json.load(af)

        self.train, self.valid = self._split(split_file)
        self.mean, self.std = self._compute_mean(meanstd_file)

    def _split(self, split_file):
        split = torch.load(split_file)
        train = split["train"]
        valid = split["valid"]
        return train, valid

    def _load_image(self, img_index, bgr=True):
        img_file = os.path.join(self.img_folder, self.mpii["annolist"][img_index]["image"])
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
            if self.single_person:
                img_ids = set(np.array(self.train)[:, 0].tolist())
            else:
                img_ids = self.train

            for index in img_ids:
                img = self._load_image(index, bgr=False)
                mean += img.reshape((-1, 3)).mean(0)
                std += img.reshape((-1, 3)).std(0)
            mean /= len(img_ids)
            std /= len(img_ids)
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

    def restore_image(self, img):
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        img = img.astype(np.float32)
        if img.ndim == 4:
            img = img.transpose(0, 2, 3, 1)
        elif img.ndim == 3:
            img = img.transpose(1, 2, 0)
        else:
            raise ValueError("Unrecognized image shape")
        return ((img + self.mean) * 255).round().clip(0, 255).astype(np.uint8)

    def _draw_label(self, points, target_map, sigma=None):
        # Generate ground truth
        kwargs = dict()
        if sigma is not None:
            kwargs['sigma'] = sigma
        for ijoint in range(points.shape[0]):
            if points[ijoint, 2] > 0:
                self.heatmap_gen(points[ijoint, :2], ijoint, target_map, **kwargs)

    def _get_item_single(self, index):
        sf = float(self.scale_factor)
        rf = float(self.rot_factor)

        if self.is_train:
            img_index, person_index = self.train[index]
        else:
            img_index, person_index = self.valid[index]

        ann = self.mpii["annolist"][img_index]["annorect"][person_index]

        # =====
        # Image

        img_bgr = self._load_image(img_index, bgr=True)

        img_size = np.array(list(img_bgr.shape[:2][::-1]), dtype=np.float32) # W, H

        center = np.array(ann["objpos"], dtype=np.float32)
        scale = ann["scale"] * 200
        rotate = 0
        flip = False

        # Image augmentation
        if self.is_train:
            scale_aug = truncnorm.rvs(-1, 1, loc=1, scale=sf)
            scale = scale * scale_aug
            rotate = truncnorm.rvs(-1, 1, loc=0, scale=rf) \
                    if np.random.rand() <= 0.6 else 0

            center_aug_off = truncnorm.rvs(-2, 2, loc=0, scale=self.trans_factor * scale, size=2)
            center += center_aug_off

            # Color distort
            img_bgr = (img_bgr.astype(np.float32) * (np.random.rand(3) * 0.4 + 0.8)).round().clip(0, 255).astype(np.uint8)

            # Flip
            if np.random.rand() <= 0.5:
                flip = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        # Prepare image and groundtruth map
        img_transform_mat = get_transform(center, None, [self.img_res[0], self.img_res[1]], rot=rotate, scale=float(self.img_res[0]) / scale)
        img_bgr_warp = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(self.img_res[0], self.img_res[1]), flags=cv2.INTER_LINEAR)
        img = img_bgr_warp[..., ::-1].astype(np.float32) / 255

        # Color normalize
        img -= self.mean

        img = img.transpose(2, 0, 1)

        # =====
        # Keypoint Map

        keypoints = np.zeros((NUM_PARTS, 3), dtype=np.float32)

        if len(ann["annopoints"]) > 0:
            for ijoint, point in ann["annopoints"].items():
                keypoints[int(ijoint)] = np.array([point[0], point[1], 2 if point[2] is None else (point[2]+1)])

        if flip:
            keypoints_tf = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))
        else:
            keypoints_tf = keypoints.copy()

        # keypoints: #person * #joints * 3
        keypoints_tf = np.c_[
                transform(keypoints_tf[..., :2], center, None, (self.kpmap_res[0], self.kpmap_res[1]), rot=rotate, scale=float(self.kpmap_res[0]) / scale),
                keypoints_tf[:, [2]]
            ]

        if not isinstance(self.kpmap_sigma, list):
            kp_map = np.zeros((NUM_PARTS, self.kpmap_res[1], self.kpmap_res[0]), dtype=np.float32)
            self._draw_label(keypoints_tf, kp_map, sigma=self.kpmap_sigma)
        else:
            kp_map = list()
            for kpmsigma in self.kpmap_sigma:
                kpm = np.zeros((NUM_PARTS, self.kpmap_res[1], self.kpmap_res[0]), dtype=np.float32)
                self._draw_label(keypoints_tf, kpm, sigma=kpmsigma)
                kp_map.append(kpm)

        result = {
            'index': index,
            'img_index': img_index,
            'person_index': person_index,
            'img': torch.from_numpy(img),
            'center': torch.from_numpy(center),
            'scale': scale,
            'keypoint_ori': torch.from_numpy(keypoints),
            'keypoint': torch.from_numpy(keypoints_tf),
            'img_transform': torch.from_numpy(img_transform_mat),
            'img_flipped': flip,
            'img_ori_size': torch.from_numpy(img_size.astype(np.int32)),
            'keypoint_map': [torch.from_numpy(kpm) for kpm in kp_map] if isinstance(kp_map, list) else torch.from_numpy(kp_map),
            'head_box': torch.tensor(ann["head_rect"], dtype=torch.float)
        }

        return result

    def _get_item_multi(self, index):
        assert False, "Method not compatible"
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
        scale = float(img_size.max())
        rotate = 0
        flip = False

        # Image augmentation
        if self.is_train:
            scale = scale * ((np.random.randn(1) * sf) + 1).clip(1-sf, 1+sf)[0]
            rotate = (np.random.randn(1) * rf).clip(-2*rf, 2*rf)[0] \
                    if np.random.rand() <= 0.6 else 0

            center += np.random.randint(-0.02 * scale, 0.02 * scale, size=2)

            # Flip
            if np.random.rand() <= 0.5:
                flip = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        # Prepare image and groundtruth map
        img_transform_mat = get_transform(center, scale, [self.img_res[0], self.img_res[1]], rot=rotate)
        img_bgr = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(self.img_res[0], self.img_res[1]), flags=cv2.INTER_LINEAR)

        img = img_bgr[..., ::-1].astype(np.float32) / 255

        if self.is_train:
            # Color distort
            img = (img * (np.random.rand(3) * 0.4 + 0.8)).clip(0, 1).astype(np.float32) 

        # Color normalize
        img -= self.mean

        img = img.transpose(2, 0, 1)

        # =====
        # Keypoint Map

        anns = self.mpii["annolist"][img_index]["annorect"]

        keypoints = np.zeros((len(anns), NUM_PARTS, 3), dtype=np.float32)

        num_person = 0
        for ann in anns:
            if len(ann["annopoints"]) > 0:
                for ijoint, point in ann["annopoints"].items():
                    keypoints[num_person, int(ijoint)] = np.array([point[0], point[1], 2 if point[2] is None else (point[2]+1)])
                num_person += 1
        keypoints = keypoints[:num_person]
        
        if flip:
            keypoints = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))

        # keypoints: #person * #joints * 3
        keypoints_tf = transform(keypoints.reshape((-1, 3))[...,:2], center, scale, [self.kpmap_res[0], self.kpmap_res[1]], rot=rotate)
        keypoints_tf = np.c_[keypoints_tf, keypoints.reshape((-1, 3))[:, 2]].reshape(keypoints.shape)
        
        target_map = np.zeros((NUM_PARTS, self.kpmap_res[1], self.kpmap_res[0]), dtype=np.float32)

        for iperson, points in enumerate(keypoints_tf):
            self._draw_label(points, target_map)

        extra = {
            'index': index,
            'center': torch.from_numpy(center),
            'scale': scale, 
            'keypoints': torch.from_numpy(keypoints) if keypoints.shape[0] > 1 else None,
            'keypoints_tf': torch.from_numpy(keypoints_tf) if keypoints.shape[0] > 1 else None
        }

        return torch.from_numpy(img), \
            torch.from_numpy(target_map), \
            extra

    def __getitem__(self, index):
        if self.single_person:
            return self._get_item_single(index)
        else:
            return self._get_item_multi(index)

    @classmethod
    def collate_function(cls, batch):
        NON_COLLATE_KEYS = ["img_flipped"]
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

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    mpii = MPII("data/mpii/images",
        "data/mpii/mpii_human_pose.json",
        "data/mpii/split_sig.pth",
        "data/mpii/mean_std.pth",
        True,
        True,
        img_res=[192, 256],
        kpmap_res=[48, 64],
        kpmap_sigma=[2.6, 2., 1.7, 1.4],
        scale_factor=0.,
        rot_factor=30,
        trans_factor=0.0)

    data_inds = np.arange(len(mpii))
    np.random.shuffle(data_inds)
    for data_index in data_inds:
        print(data_index)
        result = mpii[data_index]
        img = mpii.restore_image(result["img"])
        kpmaps = result["keypoint_map"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, kpm in enumerate(kpmaps):
            kpm_sumed = cv2.applyColorMap((kpm.numpy().max(axis=0) * 255).round().astype(np.uint8).clip(0, 255), cv2.COLORMAP_HOT)
            kpm_sumed = cv2.resize(kpm_sumed, img.shape[:2][::-1])
            draw = cv2.addWeighted(img, 0.9, kpm_sumed, 0.3, 0.)
            axes.flat[i].imshow(draw)

        plt.show()
