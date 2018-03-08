from __future__ import print_function, absolute_import
import torch
from torch.utils.data.dataloader import default_collate
import torchvision.utils as vutils

import pose.models as models
import pose.datasets as datasets
from pose.utils.evaluation import match_locate, PR_locate
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate

import cv2

from pycocotools.coco import COCO

import numpy as np
import copy

try:
    profile
except NameError:
    profile = lambda func: func

"""
1. Detect person location / head
2. Generate maps of part location and embedding
3. Input into network, output vector fields
4. Update part location
5. A: If iterating counter less than M, goto 2 (Stack Sharing/Non-sharing)
   B: If stable, goto 2 (Stack Non-sharing)

# Keypoint map fusion
- Could banlance semantic for image and keypoint maps

# Loss
- from loose to tight / equal

Input:
    A:
        image map: 128
        fuse map: 128
        keypoint map: 2 * 17
    B:
        image/fuse map: 256
        keypoint map: 2 * 17
    C:
        image/fuse map: 256
        keypoint map: 256 (after fusion)

Output:
    offset field: 2 * 17
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
            models.PoseNet(inp_dim=3, out_dim=1,
                           hg_dim=self.hparams["model"]["hg_dim"],
                           bn=self.hparams["model"]["bn"]).cuda())

        self.criterion = models.PoseMapLoss().cuda()

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
                                               locmap_res=OUT_RES,
                                               locate_res=OUT_RES,
                                               mask_res=OUT_RES)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             img_res=INP_RES,
                                             locmap_res=OUT_RES,
                                             locate_res=OUT_RES,
                                             mask_res=OUT_RES)

        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

        self.pose_mgr = models.PoseManager(max(self.hparams["train_batch"], self.hparams["test_batch"]), 1, (OUT_RES, OUT_RES), 30,
                                           cuda=False, sigma=int(self.hparams["dataset"]["label_sigma"]), filter_inside=False, gen_embedding=False)

        self.train_drop_last = True

        self.posemap_parser = models.PoseMapParser(cuda=True, threshold=self.hparams["model"]["parse_threshold"])

    def epoch(self, epoch):
        self.hparams["learning_rate"] = adjust_learning_rate(
            self.optimizer,
            epoch,
            self.hparams["learning_rate"],
            self.hparams["schedule"],
            self.hparams["lr_gamma"])
        # # decay sigma
        # label_sigma_decay = self.hparams["dataset"]["label_sigma_decay"]
        # if label_sigma_decay > 0:
        #     self.train_dataset.label_sigma *= label_sigma_decay
        #     self.val_dataset.label_sigma *= label_sigma_decayn

    def summary_image(self, img, pred, gt, pred_point, gt_point, match_pred, match_gt, indices_TP, point_factor, mask, title, step):
        TP_COLOR = (95, 162, 44)
        FP_COLOR = (0, 0, 255)
        CP_COLOR = (255, 0, 0)

        tb_num = 6
        tb_img = img[:tb_num].numpy() + self.train_dataset.mean[None, :, None, None]
        tb_gt = gt[:tb_num].numpy()
        tb_pred = pred[:tb_num].numpy()
        tb_mask = (~mask[:tb_num]).numpy()
        show_img = np.zeros((tb_num * 2, 3, INP_RES, INP_RES))
        for iimg in range(tb_num):
            cur_img = (tb_img[iimg][::-1].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            cur_gt = tb_gt[iimg].transpose(1, 2, 0) + tb_mask[iimg][:, :, None]
            cur_gt = cv2.applyColorMap(
                cv2.resize(
                    (cur_gt * 255).clip(0, 255).astype(np.uint8),
                    (INP_RES, INP_RES)),
                cv2.COLORMAP_HOT)
            cur_gt = cv2.addWeighted(cur_img, 1, cur_gt, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
            show_img[iimg] = cur_gt

            cur_pred = cv2.applyColorMap(
                cv2.resize(
                    (tb_pred[iimg].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                    (INP_RES, INP_RES)),
                cv2.COLORMAP_HOT)
            cur_pred = cv2.addWeighted(cur_img, 1, cur_pred, 0.5, 0)
            # Draw points
            cur_pred_point = pred_point[iimg]
            cur_gt_point = gt_point[iimg]
            cur_match_pred = match_pred[iimg]
            cur_match_gt = match_gt[iimg]
            cur_indices_TP = indices_TP[iimg]
            for ipoint, point in enumerate(cur_pred_point):
                pred_pt = (int(point[0] * point_factor), int(point[1] * point_factor))
                if ipoint in cur_match_pred:
                    gt_pt = cur_gt_point[cur_match_gt[cur_match_pred.tolist().index(ipoint)]]
                    gt_pt = (int(gt_pt[0] * point_factor), int(gt_pt[1] * point_factor))
                    cv2.line(cur_pred, pred_pt, gt_pt, TP_COLOR, thickness=2)
                if ipoint in cur_indices_TP:
                    color = TP_COLOR
                else:
                    color = FP_COLOR
                cv2.circle(cur_pred, pred_pt, 5, color, thickness=-1)
            for point in cur_gt_point:
                gt_pt = (int(point[0] * point_factor), int(point[1] * point_factor))
                color = CP_COLOR
                cv2.circle(cur_pred, gt_pt, 5, color, thickness=-1)

            cur_pred = cur_pred.transpose(2, 0, 1)[::-1].astype(np.float32) / 255
            show_img[tb_num + iimg] = cur_pred

        show_img = vutils.make_grid(torch.from_numpy(show_img), nrow=tb_num, range=(0, 1))
        config.tb_writer.add_image(title, show_img, step)

    def summary_histogram(self, n_iter):
        for name, param in self.model.named_parameters():
            config.tb_writer.add_histogram("locate." + name, param.clone().cpu().data.numpy(), n_iter, bins="doane")

    @profile
    def process(self, batch, train, detail=None):
        model_loc = self.model
        criterion_mse = self.criterion

        img = batch["img"]
        locate_map_gt = batch["locate_map"]
        mask = batch["mask"]
        locate_gt = batch["locate"]

        volatile = not train

        img_var = torch.autograd.Variable(img.cuda(async=True), volatile=volatile)
        locate_map_gt_var = torch.autograd.Variable(locate_map_gt.cuda(async=True), volatile=volatile)
        mask_var = torch.autograd.Variable(mask.cuda(async=True), volatile=volatile)

        for i in range(len(locate_gt)):
            if locate_gt[i] is None:
                locate_gt[i] = torch.FloatTensor(0)

        # Locate nose
        locate_map_pred_var, _ = model_loc(img_var)

        # Locating Loss
        loss = criterion_mse(locate_map_pred_var, locate_map_gt_var, mask_var)

        # Parse locate_map_pred_var, convert it into ground truth keypoints format
        # locate_pred: #batch x [#batch_i_person, 2]
        locate_pred = self.posemap_parser.parse(locate_map_pred_var, mask_var, factor=1)

        # Match locating point with person in ground truth keypoints
        # match_*: #batch x #batch_i_match_index
        match_pred, match_gt = match_locate(locate_pred, locate_gt, threshold_abandon=OUT_RES/float(self.hparams["model"]["match_threshold_factor"]))

        precision, recall, indices_TP = PR_locate(locate_pred, locate_gt, match_pred, match_gt, threshold=OUT_RES/float(self.hparams["model"]["eval_threshold_factor"]))

        if detail["summary"]:
            self.summary_image(img=img,
                               pred=locate_map_pred_var.data.cpu(),
                               gt=locate_map_gt,
                               pred_point=locate_pred,
                               gt_point=locate_gt,
                               match_pred=match_pred,
                               match_gt=match_gt,
                               indices_TP=indices_TP,
                               point_factor=4,
                               mask=mask,
                               title="locate/" + ("train" if train else "val"),
                               step=detail["epoch"] + 1)

        result = {
            "loss": loss,
            "acc": recall,
            "recall": recall,
            "prec": precision,
            "index": batch["index"],
            "pred": locate_pred
        }

        return result
