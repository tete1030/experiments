#!/usr/bin/python3
# coding=utf-8

import os
import os.path as osp
import numpy as np
import cv2

import sys
from pycocotools.coco import COCO
import torch

def generate():
    kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder',
        'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist',
        'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
    max_num_joints = 17

    coco = COCO("../data/mscoco/person_keypoints_val2017.json")
    sp_split = []
    
    for aid in coco.anns.keys():
        ann = coco.anns[aid]
        if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
            continue
        if ann['iscrowd']:
            continue
        if np.sum(ann['keypoints'][2::3]) == 0 or ann['num_keypoints'] == 0 :
            continue

        sp_split.append((ann['image_id'], aid))

    torch.save({
        "train": [],
        "valid": sp_split
    }, "sp_split_val2017.pth")

if __name__ == '__main__':
    generate()
