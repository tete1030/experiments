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
                                               inp_res=INP_RES,
                                               out_res=OUT_RES,
                                               generate_map=False)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             inp_res=INP_RES,
                                             out_res=OUT_RES,
                                             generate_map=False)

        self.train_collate_fn = self.collate_function
        self.test_collate_fn = self.collate_function

        self.pose_mgr = models.PoseManager(max(self.hparams["train_batch"], self.hparams["test_batch"]), 1, (OUT_RES, OUT_RES), 30,
                                           cuda=False, sigma=int(self.hparams["dataset"]["label_sigma"]), filter_inside=False, gen_embedding=False)

        self.train_drop_last = True

        self.posemap_parser = models.PoseMapParser(cuda=True, threshold=0.05)

    def collate_function(self, batch):
        result = datasets.COCOPose.collate_function(batch)
        keypoints = result[3]["keypoints_tf"]
        locates = list()
        for i, kp in enumerate(keypoints):
            locate = list()
            for j, person in enumerate(kp):
                labeled_part_indices = torch.nonzero(person[:, 2] > 0)
                if len(labeled_part_indices) > 0:
                    mean_pos = person[labeled_part_indices[:, 0]][:, :2].mean(dim=0)
                    if (mean_pos >= -2 * int(self.hparams["dataset"]["label_sigma"])).all() and \
                        (mean_pos < (OUT_RES + 2 * int(self.hparams["dataset"]["label_sigma"]))).all():
                        locate.append(mean_pos)
            if len(locate) > 0:
                locates.append(torch.stack(locate, 0).float())
            else:
                locates.append(torch.FloatTensor(0))
        self.pose_mgr.init_with_locate(locates)
        locate_map = self.pose_mgr.generate()
        # DataLoader doesn't support data with zero size
        for i in range(len(locates)):
            if len(locates[i]) == 0:
                locates[i] = None
        result[1] = locate_map
        result[3]["locate"] = locates

        return result

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

    def summary_image(self, img, pred, gt, mask, title, step):
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
            cur_pred = cv2.addWeighted(cur_img, 1, cur_pred, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
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

        img, locate_map_gt, mask, extra = batch
        volatile = not train

        img_var = torch.autograd.Variable(img.cuda(async=True), volatile=volatile)
        locate_map_gt_var = torch.autograd.Variable(locate_map_gt.cuda(async=True), volatile=volatile)
        mask_var = torch.autograd.Variable(mask.cuda(async=True), volatile=volatile)
        if "locate" in extra:
            locate_gt = extra["locate"]
            for i in range(len(locate_gt)):
                if locate_gt[i] is None:
                    locate_gt[i] = torch.FloatTensor(0)
        else:
            keypoint_gt = extra["keypoints_tf"]
            locate_index = self.hparams["model"]["part_locate"]
            # locate_gt: #batch x [#batch_i_valid_person x 2]
            locate_gt = []
            num_gt = 0
            for kpgt in keypoint_gt:
                index_labeled = torch.nonzero(kpgt[:, locate_index, 2] > 0)
                if len(index_labeled) > 0:
                    locate_bi = kpgt[index_labeled[:, 0]][:, locate_index, :2]
                    index_inside = torch.nonzero(((locate_bi >= 0) & (locate_bi < INP_RES)).sum(dim=-1) == 2)
                    if len(index_inside) > 0:
                        locate_gt.append(locate_bi[index_inside[:, 0]])
                        num_gt += len(index_inside)
                    else:
                        locate_gt.append(torch.FloatTensor(0))
                else:
                    locate_gt.append(torch.FloatTensor(0))

        # Locate nose
        locate_map_pred_var, _ = model_loc(img_var)

        # Locating Loss
        loss = criterion_mse(locate_map_pred_var, locate_map_gt_var, mask_var)

        # Parse locate_map_pred_var, convert it into ground truth keypoints format
        # locate_pred: #batch x [#batch_i_person, 2]
        locate_pred = self.posemap_parser.parse(locate_map_pred_var, mask_var, factor=1)

        # Match locating point with person in ground truth keypoints
        # match_*: #batch x #batch_i_match_index
        # TODO: IMPROVEMENT decrease threshold_dis gradually, or not change
        match_pred, match_gt = match_locate(locate_pred, locate_gt, threshold_abandon=OUT_RES/float(self.hparams["model"]["match_threshold_factor"]))

        precision, recall = PR_locate(locate_pred, locate_gt, match_pred, match_gt, threshold=OUT_RES/float(self.hparams["model"]["eval_threshold_factor"]))

        if ("summary" in detail and detail["summary"]):
            self.summary_image(img, locate_map_pred_var.data.cpu(), locate_map_gt, mask, "locate/" + ("train" if train else "val"), detail["epoch"] + 1)

        result = {
            "loss": loss,
            "acc": recall,
            "recall": recall,
            "prec": precision,
            "index": extra["index"],
            "pred": locate_pred
        }

        return result
