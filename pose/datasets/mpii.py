from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import scipy.misc

seg_idx = [(9,8),
           (13,14), (14,15), (12,11), (11,10),
           (3,4), (4,5), (2,1), (1,0),
           (8,13), (8,12), (13,3), (12,2),
           (7,6), (3,2)]
seg_labels = ["head",
              "l upper arm", "l lower arm", "r upper arm", "r lower arm",
              "l upper leg", "l lower leg", "r upper leg", "r lower leg",
              "l shoulder line", "r shoulder line", "l body side", "r body side",
              "thorax-pelvis", "l-r hip"]
seg_colors = [[255, 0, 0],
              [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
              [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], 
              [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]

head_seg = 0
head_size_ratio = 0.6
head_sigma_ratio = 0.5

upper_arm_segs = [1,3]
upper_arm_scale_ratio = 4
upper_arm_sigma_ratio = 0.7

lower_arm_segs = [2,4]
lower_arm_scale_ratio = 4
lower_arm_sigma_ratio = 0.5

upper_leg_segs = [5,7]
upper_leg_scale_ratio = 4
upper_leg_sigma_ratio = 0.7

lower_leg_segs = [6,8]
lower_leg_scale_ratio = 4
lower_leg_sigma_ratio = 0.3

shoulder_segs = [9,10]
shoulder_scale_ratio = 5
shoulder_sigma_ratio = 0.6

side_segs = [11,12]
side_scale_ratio = 7
side_sigma_ratio = 0.5

TP_seg = 13
TP_scale_ratio = 7
TP_sigma_ratio = 0.5

LRhip_seg = 14
LRhip_scale_ratio = 7
LRhip_sigma_ratio = 0.5

seg_ratios = [head_size_ratio] + \
             [upper_arm_scale_ratio, lower_arm_scale_ratio] * 2 + \
             [upper_leg_scale_ratio, lower_leg_scale_ratio] * 2 + \
             [shoulder_scale_ratio] * 2 + [side_scale_ratio] * 2 + \
             [TP_scale_ratio] + [LRhip_scale_ratio]

seg_sigma_ratios = [head_sigma_ratio] + \
                   [upper_arm_sigma_ratio, lower_arm_sigma_ratio] * 2 + \
                   [upper_leg_sigma_ratio, lower_leg_sigma_ratio] * 2 + \
                   [shoulder_sigma_ratio] * 2 + [side_sigma_ratio] * 2 + \
                   [TP_sigma_ratio] + [LRhip_sigma_ratio]


# disable sigma (for test)
if False:
    seg_sigma_ratios = [0.] * len(seg_sigma_ratios)

class Mpii(data.Dataset):
    LABEL_POINTS_MAP = 0
    LABEL_PARTS_MAP = 1
    LABEL_MIX_MAP = 2
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', label_data=LABEL_POINTS_MAP,
                 single_person=True, meanstd_file='./data/mpii/mean.pth.tar'):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_data = label_data
        self.single_person = single_person

        # create train/val split
        with open(jsonfile) as anno_file:   
            self.anno = json.load(anno_file)

        self.train, self.valid = [], []
        for idx, val in enumerate(self.anno):
            if val['isValidation'] == True:
                self.valid.append(idx)
            else:
                self.train.append(idx)
        self.meanstd_file = meanstd_file
        self.mean, self.std = self._compute_mean()
        self.label_type = label_type

    def _compute_mean(self):
        if isfile(self.meanstd_file):
            meanstd = torch.load(self.meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for index in self.train:
                a = self.anno[index]
                img_path = os.path.join(self.img_folder, a['img_paths'])
                img = load_image(img_path) # CxHxW
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.train)
            std /= len(self.train)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, self.meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
            
        return meanstd['mean'].numpy(), meanstd['std'].numpy()
 
    def _draw_label(self, tpts, not_annoted, scale, label_data, out_target, out_mask, **kwargs):
        # Generate ground truth
        if label_data == 'points':
            for i in range(tpts.size(0)):
                if i not in not_annoted:
                    out_target[i] = draw_labelmap(out_target[i], tpts[i], self.sigma, type=self.label_type)
        elif label_data in ('parts_visible', 'parts_all'):
            coords = tpts[:, 0:2]
            seg_ratios = kwargs['seg_ratios']
            seg_sigma_ratios = kwargs['seg_sigma_ratios']
            for isi, si in enumerate(seg_idx):
                if np.intersect1d(np.array(si), not_annoted).size > 0:
                    # TODO: #NI2 missing one solution
                    continue

                # If two points are invisible, which means the whole part is invisible, only punish false positive
                mask_value = None
                if label_data == 'parts_visible' and tpts[torch.LongTensor(si)][:, 2].sum() == 0:
                    mask_value = 2

                # Indicate that there are parts presented
                out_mask[isi][out_mask[isi] != 2] = 1

                # When parts_visible, draw head as round shape
                if isi == head_seg and label_data == 'parts_visible':
                    head_top = coords[si[0]]
                    upper_neck = coords[si[1]]
                    head_center = (head_top + upper_neck) / 2
                    head_radius = (head_top - upper_neck).norm() / 2
                    head_radius *= seg_ratios[isi]
                    head_sigma = seg_sigma_ratios[isi] * head_radius
                    out_target[isi] = draw_labelmap_ex(out_target[isi], head_center.view(1,2), head_radius, head_sigma, shape='circle', mask=out_mask[isi], mask_value=mask_value)
                else:
                    size = seg_ratios[isi] * float(self.out_res) / 200.
                    out_target[isi] = draw_labelmap_ex(out_target[isi], coords[torch.LongTensor(si)], size, seg_sigma_ratios[isi] * size, shape='pillar', mask=out_mask[isi], mask_value=mask_value)

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor

        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        img = scipy.misc.imread(img_path, mode='RGB').astype(np.float32) / 255
        img_size = torch.FloatTensor(list(img.shape[:2][::-1])) # W, H

        self_pts = torch.Tensor(a['joint_self'])
        npoints = self_pts.size(0)

        self_scale = a['scale_provided']

        person_num = a['numOtherPeople'] + 1 if not self.single_person else 1

        pts_list = torch.FloatTensor(person_num, npoints, 3)
        pts_list[0] = self_pts

        scale_list = torch.FloatTensor(person_num)
        scale_list[0] = self_scale

        if self.single_person:
            c = torch.FloatTensor(a['objpos'])
            s = self_scale

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s
                s = s * 1.25
        else:
            c = img_size / 2.
            s = 1.

            for pi, (pts_other, scale_other) in enumerate(zip(a['joint_others'], a['scale_provided_other'])):
                pts_list[pi+1] = pts_other
                scale_list[pi+1] = scale_other

        tpts_list = pts_list.clone()

        # TODO: #NI1 Need a better solution
        mask_points = torch.from_numpy((np.isclose(pts_list.numpy(), 0).sum(-1) == 3).astype(np.uint8))
        not_annoted = np.nonzero(mask_points.numpy())

        r = 0
        if self.is_train:
            s = s * torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            if not self.single_person:
                # normal distribution location with mean the image center and standard deviation img_size / 4 (so that 2\sigma == img_size/2)
                c += torch.randn(2) * img_size / 4
                outer = (c < 0 | c > img_size)
                # if fall in outside, take uniform random location for the outer axis
                c[outer] = (torch.rand(2) * img_size)[outer]
                
            # Flip
            if random.random() <= 0.5:
                img = np.fliplr(img)
                tpts_list = shufflelr(tpts_list, width=img_size[0], dataset='mpii')
                c[0] = img_size[0] - c[0]

            # Brightness
            contrast_factor = np.clip(np.random.rand() / 6. + 1., 1.5, 2.5)
            img = tune_contrast(img, contrast_factor)

            # Contrast
            brightness_factor = np.clip(np.random.rand() / 6. + 1., 1.5, 2.5)
            img = tune_brightness(img, brightness_factor)

            # Color distort
            img = (img * (np.random.rand(3) * 0.4 + 0.8)).clip(0, 1).astype(np.float32)

        # Prepare image and groundtruth map
        img = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)

        # Color normalize
        img -= self.mean

        img = torch.from_numpy(img.transpose(2, 0, 1))

        tpts_list.view(-1, 3)[:, 0:2] = torch.from_numpy(transform(tpts_list.view(-1, 3)[:, 0:2], c, s, [self.out_res, self.out_res], rot=r).T)

        is_points = (self.label_data in (Mpii.LABEL_POINTS_MAP, Mpii.LABEL_MIX_MAP))
        is_parts = (self.label_data in (Mpii.LABEL_PARTS_MAP, Mpii.LABEL_MIX_MAP))

        if is_points:
            target_points = torch.zeros(npoints, self.out_res, self.out_res)

        if is_parts:
            target_parts_visible = torch.zeros(len(seg_idx), self.out_res, self.out_res)
            target_parts_all = torch.zeros(len(seg_idx), self.out_res, self.out_res)
            mask_parts_visible = torch.zeros(len(seg_idx), self.out_res, self.out_res).byte()
            mask_parts_all = torch.zeros(len(seg_idx), self.out_res, self.out_res).byte()

        for iperson, (tpts, nann, scale) in enumerate(zip(tpts_list, not_annoted, scale_list)):
            if is_points:
                self._draw_label(tpts, nann, s, 'points', target_points, None)
            if is_parts:
                self._draw_label(tpts, nann, s, 'parts_visible', target_parts_visible, mask_parts_visible, seg_ratios=seg_ratios, seg_sigma_ratios=seg_sigma_ratios)
                self._draw_label(tpts, nann, s, 'parts_all', target_parts_all, mask_parts_all, seg_ratios=[1.] * len(seg_idx), seg_sigma_ratios=[0.2] * len(seg_idx))

        target = {}
        meta = {'index': index, 'center': c, 'scale': s, 
                'pts': pts_list, 'tpts': tpts_list}

        if is_points:
            target['points'] = target_points
            meta['mpoints'] = mask_points

        if is_parts:
            target['parts_v'] = target_parts_visible
            target['parts_a'] = target_parts_all
            meta['mparts_v'] = mask_parts_visible
            meta['mparts_a'] = mask_parts_all

        return img, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
