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
head_size_ratio = 0.8
head_sigma_ratio = 0.5

upper_arm_segs = [1,3]
upper_arm_scale_ratio = 6
upper_arm_sigma_ratio = 0.5

lower_arm_segs = [2,4]
lower_arm_scale_ratio = 4
lower_arm_sigma_ratio = 0.5

upper_leg_segs = [5,7]
upper_leg_scale_ratio = 6
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
TP_scale_ratio = 10
TP_sigma_ratio = 1.

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
    def __init__(self, jsonfile, img_folder, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', label_data=LABEL_POINTS_MAP,
                 meanstd_file='./data/mpii/mean.pth.tar'):
        self.img_folder = img_folder    # root image folders
        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_data = label_data

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
            
        return meanstd['mean'], meanstd['std']

    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            a = self.anno[self.train[index]]
        else:
            a = self.anno[self.valid[index]]

        img_path = os.path.join(self.img_folder, a['img_paths'])
        pts = torch.Tensor(a['joint_self'])
        # TODO: #NI1 Need a better solution
        not_annoted = np.nonzero(np.isclose(to_numpy(pts), 0).sum(1) == 3)
        # pts[:, 0:2] -= 1  # Convert pts to zero based

        # c = torch.Tensor(a['objpos']) - 1
        c = torch.Tensor(a['objpos'])
        s = a['scale_provided']

        # Adjust center/scale slightly to avoid cropping limbs
        if c[0] != -1:
            c[1] = c[1] + 15 * s
            s = s * 1.25

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img_path)  # CxHxW

        r = 0
        tpts = pts.clone()
        if self.is_train:
            s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                tpts = shufflelr(tpts, width=img.size(2), dataset='mpii')
                c[0] = img.size(2) - c[0]

            # Color
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)
        inp = color_normalize(inp, self.mean, self.std)

        # Generate ground truth
        if self.label_data == Mpii.LABEL_POINTS_MAP:
            target = torch.zeros(nparts, self.out_res, self.out_res)
            for i in range(nparts):
                # if tpts[i, 2] > 0: # This is evil!!
                if i not in not_annoted:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r))
                    target[i] = draw_labelmap(target[i], tpts[i], self.sigma, type=self.label_type)
        elif self.label_data == Mpii.LABEL_PARTS_MAP:
            target = torch.zeros(len(seg_idx), self.out_res, self.out_res)
            tpts[:, 0:2] = to_torch(transform(tpts[:, 0:2], c, s, [self.out_res, self.out_res], rot=r).T)
            coords = tpts[:, 0:2]
            for isi, si in enumerate(seg_idx):
                if np.intersect1d(np.array(si), not_annoted).size > 0:
                    # TODO: #NI2 missing one solution
                    # print("missing one")
                    # print(si)
                    continue
                if isi == head_seg:
                    head_top = coords[si[0]]
                    upper_neck = coords[si[1]]
                    head_center = (head_top + upper_neck) / 2
                    head_radius = (head_top - upper_neck).norm() / 2
                    head_radius *= seg_ratios[isi]
                    head_sigma = seg_sigma_ratios[isi] * head_radius
                    target[isi] = draw_labelmap_ex(target[isi], head_center.view(1,2), head_radius, head_sigma, shape='circle')
                else:
                    size = seg_ratios[isi] * float(self.out_res) / 200.
                    target[isi] = draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], size, seg_sigma_ratios[isi] * size, shape='pillar')

                # elif isi in arm_segs:
                #     arm_size = arm_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], arm_size, arm_sigma_ratio*arm_size, shape='pillar')
                # elif isi in leg_segs:
                #     leg_size = leg_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], leg_size, leg_sigma_ratio*leg_size, shape='pillar')
                # elif isi in shoulder_segs:
                #     shoulder_size = shoulder_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], shoulder_size, shoulder_sigma_ratio*shoulder_size, shape='pillar')
                # elif isi in side_segs:
                #     side_size = side_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], side_size, side_sigma_ratio*side_size, shape='pillar')
                # elif isi == TP_seg:
                #     TP_size = TP_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], TP_size, TP_sigma_ratio*TP_size, shape='pillar')
                # elif isi == LRhip_seg:
                #     LRhip_size = LRhip_scale_ratio * (self.inp_res / 200)
                #     draw_labelmap_ex(target[isi], coords[torch.LongTensor(si)], LRhip_size, LRhip_sigma_ratio*LRhip_size, shape='pillar')

        # Meta info
        meta = {'index' : index, 'center' : c, 'scale' : s, 
        'pts' : pts, 'tpts' : tpts}
        return inp, target, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
