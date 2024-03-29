#!python3

import os
import copy
import cv2
import numpy as np
from scipy.stats import truncnorm

import torch
import torch.utils.data as data

import matplotlib.pyplot as plt

from lib.utils.imutils import HeatmapGenerator
from lib.utils.transforms import fliplr_chwimg, fliplr_pts, get_transform, transform
from pycocotools.coco import COCO

from utils.globals import config

# FLIP_INDEX is an involution map
FLIP_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

PART_LABELS = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r',
               'sho_l', 'sho_r', 'elb_l', 'elb_r', 'wri_l', 'wri_r',
               'hip_l', 'hip_r', 'kne_l', 'kne_r', 'ank_l', 'ank_r']

PART_CONNECT = [(0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

NUM_PARTS = 17

FACTOR = 4

class COCOPose(data.Dataset):
    def __init__(self, img_folder, anno, split_file, meanstd_file,
                 train, single_person,
                 img_res=[256], minus_mean=True,
                 kpmap_res=64, locmap_res=0, mask_res=0,
                 kpmap_select=None, kpmap_sigma=1, locmap_min_sigma=0.5,
                 keypoint_res=0, keypoint_label_outsider=False, keypoint_filter=False, locate_res=0,
                 keypoint_extender=None,
                 scale_factor=0.25, rot_factor=30, person_random_selection=False,
                 custom_generator=None):
        assert not single_person # not supported in this class
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.single_person = single_person
        self.img_res = img_res
        self.minus_mean = minus_mean
        self.kpmap_res = kpmap_res
        self.locmap_res = locmap_res
        self.keypoint_extender = keypoint_extender
        self.mask_res = mask_res
        self.kpmap_select = kpmap_select
        self.kpmap_sigma = kpmap_sigma
        self.locmap_min_sigma = locmap_min_sigma
        self.keypoint_res = keypoint_res
        self.keypoint_label_outsider = keypoint_label_outsider
        self.keypoint_filter = keypoint_filter
        self.locate_res = locate_res
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.person_random_selection = person_random_selection
        self.custom_generator = custom_generator
        if self.person_random_selection:
            assert not self.single_person

        self.heatmap_gen = HeatmapGenerator(self.kpmap_res, self.kpmap_sigma)

        # create train/val split
        if isinstance(anno, COCO):
            self.coco = anno
        else:
            self.coco = COCO(anno)

        self.train, self.valid = self._split(split_file)
        if self.minus_mean:
            self.mean, self.std = self._compute_mean(meanstd_file)
        else:
            self.mean, self.std = None, None

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
            self.heatmap_gen(points, 0, target_map, sigma=sigma, out_res=out_res, normalize_factor=None)
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

    def restore_image(self, img):
        if not isinstance(img, np.ndarray):
            img = img.numpy()
        return ((img.astype(np.float32).transpose(0, 2, 3, 1) + self.mean) * 255).round().clip(0, 255).astype(np.uint8)

    def get_index_by_imgidx(self, imgidx):
        try:
            if self.is_train:
                return self.train.index(imgidx)
            else:
                return self.valid.index(imgidx)
        except ValueError:
            return -1

    def __getitem__(self, index):
        if "OPENCV_CUDA_DEVICE" in os.environ:
            CVD_ori = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if CVD_ori:
                os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["OPENCV_CUDA_DEVICE"]

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
        scale = float(img_size.max())
        rotate = 0
        flip_status = False

        # Image augmentation
        if self.is_train:
            scale = scale * ((np.random.randn(1) * sf) + 1).clip(1-sf, 1+sf)[0]
            rotate = (np.random.randn(1) * rf).clip(-2*rf, 2*rf)[0] \
                    if np.random.rand() <= 0.6 else 0

            center += np.random.randint(-0.02 * scale, 0.02 * scale, size=2)

            # Color distort
            img_bgr = (img_bgr.astype(np.float32) * (np.random.rand(3) * 0.4 + 0.8)).round().clip(0, 255).astype(np.uint8)

            # Flip
            if np.random.rand() <= 0.5:
                flip_status = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        # Prepare image and groundtruth map
        img_list = list()
        img_transform_list = list()
        for cur_res in (self.img_res if isinstance(self.img_res, list) else [self.img_res]):
            img_transform_mat = get_transform(center, scale, [cur_res, cur_res], rot=rotate)
            img_transform_list.append(img_transform_mat)
            img_bgr_warp = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(cur_res, cur_res), flags=cv2.INTER_LINEAR)
            img = img_bgr_warp[..., ::-1].astype(np.float32) / 255
            # Color normalize
            if self.minus_mean:
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

        num_person = keypoints.shape[0]

        if flip_status:
            keypoints_tf = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))
        else:
            keypoints_tf = keypoints.copy()

        # keypoints: #person * #joints * 3
        keypoints_tf_ids = np.arange(num_person, dtype=np.int32)
        keypoints_tf = np.c_[
                transform(keypoints_tf.reshape((-1, 3))[...,:2], center, scale, [self.kpmap_res, self.kpmap_res], rot=rotate),
                keypoints_tf.reshape((-1, 3))[:, 2]
            ].reshape(keypoints_tf.shape)

        if self.keypoint_label_outsider:
            outsider_mask = ((keypoints_tf[:, :, 0] < 0) | (keypoints_tf[:, :, 0] >= self.kpmap_res) | \
                            (keypoints_tf[:, :, 1] < 0) | (keypoints_tf[:, :, 1] >= self.kpmap_res))
            keypoints_tf[outsider_mask.nonzero() + (2,)] = 0

        if self.keypoint_filter:
            sel_person_filter = (keypoints_tf[:, :, 2].sum(axis=1) > 0).nonzero()
            keypoints_tf = keypoints_tf[sel_person_filter]
            keypoints_tf_ids = keypoints_tf_ids[sel_person_filter]

        if self.person_random_selection and keypoints_tf.shape[0] > 1:
            person_indices = np.arange(keypoints_tf.shape[0])
            np.random.shuffle(person_indices)
            sel_person_random = person_indices[:np.random.randint(low=1, high=person_indices.shape[0]+1)]
            keypoints_tf = keypoints_tf[sel_person_random]
            keypoints_tf_ids = keypoints_tf_ids[sel_person_random]

        locate_mean, locate_std, locate_in_kp = self._compute_locate_mean_std(keypoints_tf)
        locate_mean_ret, locate_std_ret = copy.deepcopy(locate_mean), copy.deepcopy(locate_std)
        if self.locate_res > 0 and self.locate_res != self.kpmap_res:
            assert locate_mean_ret.dtype == np.float32
            locate_mean_ret = locate_mean_ret * (float(self.locate_res) / self.kpmap_res)
            locate_std_ret = [(lsr * (float(self.locate_res) / self.kpmap_res)) if lsr is not None else None for lsr in locate_std_ret]

        if self.keypoint_extender:
            keypoints_tf = self.keypoint_extender(keypoints_tf)

        keypoints_tf_ret = keypoints_tf.copy()
        if self.keypoint_res > 0 and self.keypoint_res != self.kpmap_res:
            assert keypoints_tf_ret.dtype == np.float32
            keypoints_tf_ret[:, :, :2] = keypoints_tf_ret[:, :, :2] * (float(self.keypoint_res) / self.kpmap_res)

        if self.kpmap_select is None:
            draw_parts = []
        elif self.kpmap_select == "all":
            assert self.kpmap_res > 0
            draw_parts = list(range(NUM_PARTS))
        elif self.kpmap_select == "all_ex":
            assert self.kpmap_res > 0
            draw_parts = list(range(keypoints_tf.shape[1]))
        elif isinstance(self.kpmap_select, list):
            assert self.kpmap_res > 0
            draw_parts = self.kpmap_select
        else:
            raise ValueError("Wrong kpmap_select")

        kp_map = None
        if len(draw_parts) > 0:
            kp_map = np.zeros((len(draw_parts), self.kpmap_res, self.kpmap_res), dtype=np.float32)
            for points in keypoints_tf:
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

        if self.custom_generator:
            custom = self.custom_generator(img, keypoints_tf_ret, mask_noncrowd)

        result = {
            "index": index,
            "img_index": img_index,
            "img": [torch.from_numpy(img) for img in img_list] if isinstance(self.img_res, list) else torch.from_numpy(img_list[0]),
            "center": torch.from_numpy(center),
            "scale": scale,
            "keypoint_ori": torch.from_numpy(keypoints) if keypoints.shape[0] > 0 else None,
            "keypoint": torch.from_numpy(keypoints_tf_ret) if keypoints_tf_ret.shape[0] > 0 else None,
            "keypoint_ids": torch.from_numpy(keypoints_tf_ids).long() if keypoints_tf_ids.shape[0] > 0 else None,
        }

        result["img_transform"] = [torch.from_numpy(mat) for mat in img_transform_list] if isinstance(self.img_res, list) else torch.from_numpy(img_transform_list[0])
        result["img_flipped"] = flip_status
        result["img_ori_size"] = torch.from_numpy(img_size.astype(np.int32))

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

        if self.custom_generator:
            result["custom"] = custom

        if "OPENCV_CUDA_DEVICE" in os.environ:
            if CVD_ori:
                os.environ["CUDA_VISIBLE_DEVICES"] = CVD_ori

        return result

    @classmethod
    def collate_function(cls, batch):
        NON_COLLATE_KEYS = ["keypoint_ori", "keypoint", "keypoint_ids", "locate", "locate_std", "locate_in_kp", "img_transform", "img_flipped"]
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

class COCOSinglePose(data.Dataset):
    def __init__(self, img_folder, anno, split_file, meanstd_file,
                 is_train, img_res=(192, 256), minus_mean=True,
                 ext_border=(0., 0.),
                 kpmap_res=(48, 64), keypoint_res=(48, 64),
                 kpmap_sigma=1, scale_factor=0.25, rot_factor=30, trans_factor=0.05,
                 half_body_prob=0., half_body_num_joints=17,
                 preserve_transform_data=False):
        """COCO Keypoints Single Person
        
        Arguments:
            img_folder {str} -- Image Folder
            anno {COCO} -- COCO object
            split_file {str} -- Split file
            meanstd_file {str} -- Mean and Std store file
            is_train {bool} -- Training mode indicator
        
        Keyword Arguments:
            img_res {tuple of int} -- (W, H) (default: {(192, 256)})
            minus_mean {bool} -- if mean should be deducted (default: {True})
            ext_border {tuple of float} -- border extended scale (default: {(0., 0.)})
            kpmap_res {tuple of int or None} -- (W, H) (default: {(48, 64)})
            keypoint_res {tuple of int or None} -- (W, H) (default: {None})
            kpmap_sigma {float} -- gaussian kernel size (default: {1})
            scale_factor {float} -- data augmentation scale (default: {0.25})
            rot_factor {float} -- data augmentation rotation (default: {30})
            trans_factor {float} -- data augmentation translation (default: {0.05})
        """

        self.img_folder = img_folder    # root image folders
        self.is_train = is_train           # training set or test set
        self.img_res = tuple(img_res)
        self.minus_mean = minus_mean
        self.ext_border = tuple(ext_border) if ext_border else None
        self.kpmap_res = tuple(kpmap_res) if kpmap_res else None
        assert isinstance(kpmap_sigma, (int, float, list))
        self.kpmap_sigma = kpmap_sigma
        self.keypoint_res = tuple(keypoint_res)
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.trans_factor = trans_factor
        self.resize_scale = 1
        self.preserve_transform_data = preserve_transform_data

        if self.kpmap_res is not None:
            assert float(self.img_res[0]) / float(self.img_res[1]) == float(self.kpmap_res[0]) / float(self.kpmap_res[1])
        
        assert float(self.img_res[0]) / float(self.img_res[1]) == float(self.keypoint_res[0]) / float(self.keypoint_res[1])

        self.heatmap_gen = HeatmapGenerator()

        # create train/val split
        if isinstance(anno, COCO):
            self.coco = anno
        else:
            self.coco = COCO(anno)

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)
        self.num_joints_half_body = half_body_num_joints
        self.prob_half_body = half_body_prob

        self.train, self.valid = self._split(split_file)
        if self.minus_mean:
            self.mean, self.std = self._compute_mean(meanstd_file)

    def _split(self, split_file):
        if split_file is not None and os.path.isfile(split_file):
            split = torch.load(split_file)
            train = split["train"]
            valid = split["valid"]
        else:
            raise ValueError("%s not found" % (split_file,))

        return train, valid

    def _compute_mean(self, meanstd_file):
        if meanstd_file is None:
            print("Warning: not using mean_std")
            return np.array([0.]*3), np.array([1.]*3)

        if os.path.isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            raise ValueError("meanstd_file does not exist")

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

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def _load_image(self, img_index, bgr=True):
        img_info = self.coco.loadImgs(img_index)[0]
        path = img_info['file_name']
        img_file = os.path.join(self.img_folder, path)
        img_bgr = cv2.imread(img_file)
        return img_bgr if bgr else img_bgr[..., ::-1]

    def _draw_label(self, points, target_map, out_res=None, sigma=None):
        # Generate ground truth
        kwargs = dict()
        if sigma is not None:
            kwargs['sigma'] = sigma
        if out_res is not None:
            kwargs['out_res'] = out_res
        for ijoint in range(points.shape[0]):
            if points[ijoint, 2] > 0:
                self.heatmap_gen(points[ijoint, :2], ijoint, target_map, **kwargs)

    def set_resize_scale(self, resize_scale=1):
        img_res = (
            int(np.ceil(self.img_res[0] * resize_scale / FACTOR)) * FACTOR,
            int(np.ceil(self.img_res[1] * resize_scale / FACTOR)) * FACTOR)
        if self.kpmap_res is not None:
            assert self.img_res[0] / self.kpmap_res[0] == FACTOR and self.img_res[1] / self.kpmap_res[1] == FACTOR
            kpmap_res = (
                int(np.ceil(self.kpmap_res[0] * resize_scale)),
                int(np.ceil(self.kpmap_res[1] * resize_scale)))
            assert img_res[0] / kpmap_res[0] == FACTOR and img_res[1] / kpmap_res[1] == FACTOR
        if self.keypoint_res is not None:
            assert self.img_res[0] / self.keypoint_res[0] == FACTOR and self.img_res[1] / self.keypoint_res[1] == FACTOR
            keypoint_res = (
                int(np.ceil(self.keypoint_res[0] * resize_scale)),
                int(np.ceil(self.keypoint_res[1] * resize_scale)))
            assert img_res[0] / keypoint_res[0] == FACTOR and img_res[1] / keypoint_res[1] == FACTOR
        self.resize_scale = resize_scale

    def half_body_transform(self, keypoints):
        upper_keypoints = []
        lower_keypoints = []
        for part_id in range(NUM_PARTS):
            if keypoints[part_id, 2] > 0:
                if part_id in self.upper_body_ids:
                    upper_keypoints.append(keypoints[part_id, :2])
                else:
                    lower_keypoints.append(keypoints[part_id, :2])

        if np.random.randn() < 0.5 and len(upper_keypoints) > 2:
            selected_keypoints = upper_keypoints
        else:
            selected_keypoints = lower_keypoints \
                if len(lower_keypoints) > 2 else upper_keypoints

        if len(selected_keypoints) < 2:
            return None

        selected_keypoints = np.array(selected_keypoints, dtype=np.float32)
        center = selected_keypoints.mean(axis=0)[:2]

        left_top = np.amin(selected_keypoints, axis=0)[:2]
        right_bottom = np.amax(selected_keypoints, axis=0)[:2]

        w = (right_bottom[0] - left_top[0]) * 1.3
        h = (right_bottom[1] - left_top[1]) * 1.3

        return np.array([center[0] - w  / 2, center[1] - h / 2, w, h], dtype=np.float32)

    def get_transformed_image(self, img_bgr, img_res, center=None, rotate=None, scale=None, mat=None):
        img_transform_mat = get_transform(center, None, (img_res[0], img_res[1]), rot=rotate, scale=scale)
        if mat is not None:
            img_transform_mat = np.dot(mat, img_transform_mat)
        img_bgr_warp = cv2.warpAffine(img_bgr, img_transform_mat[:2], dsize=(img_res[0], img_res[1]), flags=cv2.INTER_LINEAR)
        img = img_bgr_warp[..., ::-1].astype(np.float32) / 255

        if self.minus_mean:
            img -= self.mean
        img = img.transpose(2, 0, 1)
        return img, img_bgr_warp, img_transform_mat

    def __getitem__(self, index):
        sf = float(self.scale_factor)
        rf = float(self.rot_factor)

        if self.is_train:
            img_index, ann_index = self.train[index]
        else:
            img_index, ann_index = self.valid[index]

        if self.is_train and self.resize_scale != 1:
            # assume the factor
            img_res = (
                int(np.ceil(self.img_res[0] * self.resize_scale / FACTOR)) * FACTOR,
                int(np.ceil(self.img_res[1] * self.resize_scale / FACTOR)) * FACTOR)
            if self.kpmap_res is not None:
                kpmap_res = (
                    int(np.ceil(self.kpmap_res[0] * self.resize_scale)),
                    int(np.ceil(self.kpmap_res[1] * self.resize_scale)))
            else:
                kpmap_res = None
            if isinstance(self.kpmap_sigma, list):
                kpmap_sigma = list()
                for kpmsig in self.kpmap_sigma:
                    kpmap_sigma.append(kpmsig * self.resize_scale)
            else:
                kpmap_sigma = self.kpmap_sigma * self.resize_scale
            keypoint_res = (
                int(np.ceil(self.keypoint_res[0] * self.resize_scale)),
                int(np.ceil(self.keypoint_res[1] * self.resize_scale)))
        else:
            img_res = self.img_res
            kpmap_res = self.kpmap_res
            kpmap_sigma = self.kpmap_sigma
            keypoint_res = self.keypoint_res

        img_bgr = self._load_image(img_index, bgr=True)
        ann = self.coco.anns[ann_index]

        img_size = np.array(list(img_bgr.shape[:2][::-1]), dtype=np.float32) # W, H
        img_res_np = np.array(img_res, dtype=np.float32)
        bbox = np.array(ann['bbox']).reshape(4).astype(np.float32)
        keypoints = np.array(ann["keypoints"], dtype=np.float32).reshape((NUM_PARTS, 3))
        half_body = False
        if self.is_train:
            if (np.sum(keypoints[:, 2]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                new_bbox = self.half_body_transform(keypoints)
                if new_bbox is not None:
                    half_body = True
                    bbox = new_bbox
        crop_size = bbox[2:] * (1 + np.array(self.ext_border, dtype=np.float32))

        center = bbox[:2] + bbox[2:] / 2

        crop_ratio = crop_size / img_res_np
        if crop_ratio[0] > crop_ratio[1]:
            bbox_size = crop_size[0]
            arg_min_shape = 0
        else:
            bbox_size = crop_size[1]
            arg_min_shape = 1
        min_crop_size = crop_size.min()
        rotate = 0
        flip_status = False

        if self.is_train:
            if self.resize_scale != 1:
                # continuous scale
                bbox_size *= self.resize_scale / (img_res[0] / self.img_res[0])

            scale_aug = truncnorm.rvs(-1, 1, loc=1, scale=sf)
            bbox_size = bbox_size * scale_aug
            rotate = truncnorm.rvs(-1, 1, loc=0, scale=rf) \
                    if np.random.rand() <= 0.6 else 0

            center_aug_off = truncnorm.rvs(-2, 2, loc=0, scale=self.trans_factor * min_crop_size, size=2)
            center += center_aug_off

            # Color distort
            img_bgr = (img_bgr.astype(np.float32) * (np.random.rand(3) * 0.4 + 0.8)).round().clip(0, 255).astype(np.uint8)

            # Flip
            if np.random.rand() <= 0.5:
                flip_status = True
                img_bgr = cv2.flip(img_bgr, 1)
                center[0] = img_size[0] - center[0]

        img_scale = float(img_res[arg_min_shape]) / bbox_size
        img, img_bgr_warp, img_transform_mat = self.get_transformed_image(img_bgr, img_res, center=center, rotate=rotate, scale=img_scale)

        if flip_status:
            keypoints_tf = fliplr_pts(keypoints, FLIP_INDEX, width=int(img_size[0]))
        else:
            keypoints_tf = keypoints.copy()
        
        keypoints_tf = np.c_[
                transform(keypoints_tf[:, :2], center, None, (keypoint_res[0], keypoint_res[1]), rot=rotate, scale=float(keypoint_res[arg_min_shape]) / bbox_size),
                keypoints_tf[:, [2]]
            ]

        if kpmap_res is not None:
            keypoints_map_tf = keypoints_tf.copy()
            if kpmap_res != keypoint_res:
                keypoints_map_tf[..., :2] = keypoints_map_tf[..., :2] * (float(kpmap_res[0]) / keypoint_res[0])

            if not isinstance(kpmap_sigma, list):
                kp_map = np.zeros((NUM_PARTS, kpmap_res[1], kpmap_res[0]), dtype=np.float32)
                self._draw_label(keypoints_map_tf, kp_map, out_res=kpmap_res, sigma=kpmap_sigma)
            else:
                kp_map = list()
                for kpmsigma in kpmap_sigma:
                    kpm = np.zeros((NUM_PARTS, kpmap_res[1], kpmap_res[0]), dtype=np.float32)
                    self._draw_label(keypoints_map_tf, kpm, out_res=kpmap_res, sigma=kpmsigma)
                    kp_map.append(kpm)
        else:
            kp_map = None

        if config.vis and False:
            if self.is_train:
                # print("Aug Setting: scale_fac %.2f , rotate_fac %.2f, trans_fac %.2f" % (sf, rf, self.trans_factor))
                print("Aug: rescale %.2f , rot %.2f , center_off (%.2f, %.2f), flip %s, half %s" % (
                    scale_aug, rotate, center_aug_off[0], center_aug_off[1],
                    "Y" if flip_status else "N",
                    "Y" if half_body else "N"))

            if flip_status:
                bbox[0] = img_size[0] - bbox[0] - bbox[2]
            cv2.rectangle(img_bgr, (bbox[0], bbox[1]), ((bbox[0] + bbox[2]), (bbox[1] + bbox[3])), (255, 0, 0), thickness=2)

            plt.figure()
            plt.imshow(img_bgr[..., ::-1])
            plt.title("Original Image")
            plt.figure()
            plt.imshow(img_bgr_warp[..., ::-1])
            plt.title("Cropped Image")

            if not isinstance(kp_map, list):
                kp_map_list = [kp_map]
            else:
                kp_map_list = kp_map
            img_small = cv2.resize(img_bgr_warp, (kp_map_list[0].shape[2], kp_map_list[0].shape[1]))
            for i, kpm in enumerate([kp_map_list[-1]]):
                fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(18,12), gridspec_kw={"hspace": 0, "wspace": 0})
                for iax in range(axes.size):
                    ax = axes.flat[iax]
                    ax.tick_params(bottom="off", left="off", labelbottom="off", labelleft="off")
                    if iax < kpm.shape[0]:
                        ax.imshow(cv2.addWeighted(img_small, 1.0, cv2.applyColorMap((kpm[iax].clip(0., 1.) * 255).round().astype(np.uint8), cv2.COLORMAP_HOT), 0.4, 0)[..., ::-1])
                    else:
                        ax.axis("off")
                fig.suptitle("kernel: %d x %d" % (kpmap_sigma[i], kpmap_sigma[i]))

            plt.show()

        result = {
            "index": index,
            "img_index": img_index,
            "ann_index": ann_index,
            "img": torch.from_numpy(img),
            "keypoint_ori": torch.from_numpy(keypoints),
            "keypoint": torch.from_numpy(keypoints_tf),
            "img_transform": torch.from_numpy(img_transform_mat),
            "img_flipped": flip_status,
            "img_ori_size": torch.from_numpy(img_size.astype(np.int32))
        }
        if isinstance(kp_map, list):
            result["keypoint_map"] = [torch.from_numpy(kpm) for kpm in kp_map]
        elif isinstance(kp_map, np.ndarray):
            result["keypoint_map"] = torch.from_numpy(kp_map)
        elif kp_map is None:
            pass
        else:
            raise ValueError()

        if self.preserve_transform_data:
            result.update({
                "center": center,
                "bbox_size": bbox_size,
                "img_bgr": img_bgr,
                "img_res": img_res,
                "img_scale": img_scale,
                "img_rotate": rotate,
                "min_crop_size": min_crop_size
            })

        return result

    @classmethod
    def collate_function(cls, batch):
        NON_COLLATE_KEYS = ["img_flipped"]
        IGNORE_KEYS = ["img_bgr", "img_res", "img_scale", "img_rotate", "bbox_size", "center", "min_crop_size"]
        collate_fn = data.dataloader.default_collate
        all_keys = batch[0].keys()
        result = dict()
        for k in all_keys:
            if k in IGNORE_KEYS:
                continue
            if k in NON_COLLATE_KEYS:
                result[k] = [sample[k] for sample in batch]
            else:
                result[k] = collate_fn([sample[k] for sample in batch])
        return result
