from __future__ import print_function, absolute_import
import torch
from torch.utils.data.dataloader import default_collate
import torchvision.utils as vutils

import pose.models as models
import pose.datasets as datasets
from pose.utils.evaluation import PR_multi
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate

from pycocotools.coco import COCO

import matplotlib as mpl
import numpy as np
import scipy.ndimage as ndimage
import copy
import cv2

try:
    profile
except NameError:
    profile = lambda func: func

"""
Keypoints (17)
Direction ()

- eye-eyecenter x 2
- ear-earcenter x 2
- 

"""


FACTOR = 4
INP_RES = 256
OUT_RES = 64

class Experiment(object):
    """Stretch Experiment
    """

    def __init__(self, hparams):
        self.hparams = hparams
        self.num_parts = datasets.mscoco.NUM_PARTS

        self.model = torch.nn.DataParallel(
            models.MergeResNet(inp_size=self.num_parts*2, num_classes=self.hparams["model"]["max_person"]*self.num_parts*2).cuda()
        )

        self.criterion = models.PoseDisLoss().cuda()

        self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                          lr=self.hparams["learning_rate"],
                                          weight_decay=self.hparams["weight_decay"])

        # Only used when train and valid dataset are all from train2014
        self.coco = COCO("data/mscoco/person_keypoints_train2014.json")

        self.train_dataset = datasets.COCOPose("data/mscoco/images",
                                               self.coco,
                                               "data/mscoco/split.pth",
                                               "data/mscoco/mean_std.pth",
                                               train=True,
                                               single_person=False,
                                               img_res=INP_RES,
                                               mask_res=INP_RES,
                                               keypoint_res=INP_RES,
                                               locate_res=INP_RES)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             img_res=INP_RES,
                                             mask_res=INP_RES,
                                             keypoint_res=INP_RES,
                                             locate_res=INP_RES)


        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

    def epoch(self, epoch):
        self.hparams["learning_rate"] = adjust_learning_rate(
            self.optimizer,
            epoch,
            self.hparams["learning_rate"],
            self.hparams["schedule"],
            self.hparams["lr_gamma"])

