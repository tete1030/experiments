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

    coco = COCO("../data/mscoco/person_keypoints_train2014.json")

    split = torch.load("../data/mscoco/split.pth")

    train_split = split["train"]
    valid_split = split["valid"]

    train_sp_split = []
    valid_sp_split = []
    
    for aid in coco.anns.keys():
        ann = coco.anns[aid]
        if ann['image_id'] not in coco.imgs or ann['image_id'] == '366379':
            continue
        if ann['iscrowd']:
            continue
        if np.sum(ann['keypoints'][2::3]) == 0 or ann['num_keypoints'] == 0 :
            continue

        if ann['image_id'] in train_split:
            train_sp_split.append((ann['image_id'], aid))
        elif ann['image_id'] in valid_split:
            valid_sp_split.append((ann['image_id'], aid))
        else:
            print("Warning: %d not belonging to either" % (ann['image_id']))

    torch.save({
        "train": train_sp_split,
        "valid": valid_sp_split
    }, "sp_split.pth")

if __name__ == '__main__':
    generate()