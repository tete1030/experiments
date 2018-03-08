from __future__ import print_function, absolute_import
import torch
from torch.utils.data.dataloader import default_collate

import torchvision.utils as vutils

import pose.models as models
import pose.datasets as datasets
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate

from pycocotools.coco import COCO

import numpy as np
import cv2

try:
    profile
except NameError:
    profile = lambda func: func

"""
1. Locate
2. Light up location randomly, training generating of corresponding keypoints
3. Eval: group farest persons
4. backward each keypoint w.r.t. input locate map
"""


FACTOR = 4
INP_RES = 512
OUT_RES = 128

class Experiment(object):
    def __init__(self, hparams):
        self.hparams = hparams
        self.num_parts = datasets.mscoco.NUM_PARTS

        self.model = torch.nn.DataParallel(
            models.PoseNet(inp_dim=1, merge_inp_dim=3, out_dim=self.num_parts,
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
                                               locmap_res=INP_RES,
                                               kpmap_res=OUT_RES,
                                               kpmap_select="all",
                                               mask_res=OUT_RES,
                                               keypoint_res=OUT_RES,
                                               locate_res=INP_RES,
                                               kpmap_sigma=2,
                                               locmap_min_sigma=1,
                                               random_selection=True)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             img_res=INP_RES,
                                             locmap_res=INP_RES,
                                             kpmap_res=OUT_RES,
                                             kpmap_select="all",
                                             mask_res=OUT_RES,
                                             keypoint_res=OUT_RES,
                                             locate_res=INP_RES,
                                             kpmap_sigma=2,
                                             locmap_min_sigma=1,
                                             random_selection=True)

        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

        self.step_process = True

        self.posemap_parser = models.PoseMapParser(cuda=True, threshold=self.hparams["model"]["parse_threshold"])

    def epoch(self, epoch):
        self.hparams["learning_rate"] = adjust_learning_rate(
            self.optimizer,
            epoch,
            self.hparams["learning_rate"],
            self.hparams["schedule"],
            self.hparams["lr_gamma"])

    def summary_image(self, img, locate_map, pred_map, keypoint_gt, matches, title, step):
        MATCH_COLOR = (95, 162, 44)
        PRED_COLOR = (0, 0, 255)
        GT_COLOR = (255, 0, 0)
        tb_num = 6
        tb_img = (img[:tb_num].numpy() + self.train_dataset.mean[None, :, None, None])[:, ::-1]
        tb_locate_map = locate_map[:tb_num].numpy()
        tb_pred_map = pred_map[:tb_num].numpy()

        ncols = self.num_parts + 1
        whole_img = np.zeros((tb_num, ncols, 3, INP_RES, INP_RES))

        for i in range(len(tb_img)):
            cur_img = (tb_img[i].transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8)
            cur_locate_img = cur_img.copy()
            cur_locate = cv2.applyColorMap(
                cv2.resize(
                    (tb_locate_map[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                    (INP_RES, INP_RES)),
                cv2.COLORMAP_HOT)
            matches_dict = dict()
            for (part_idx, pos_x, pos_y), (loc_x, loc_y) in matches[i]:
                if part_idx not in matches_dict:
                    matches_dict[part_idx] = list()
                matches_dict[part_idx].append(((pos_x, pos_y), (loc_x, loc_y)))
                cv2.line(cur_locate_img, (pos_x, pos_y), (loc_x, loc_y), MATCH_COLOR, thickness=1)
                cv2.circle(cur_locate_img, (pos_x, pos_y), 2, PRED_COLOR, thickness=-1)
                cv2.circle(cur_locate_img, (loc_x, loc_y), 2, MATCH_COLOR, thickness=-1)
            cur_locate = cv2.addWeighted(cur_locate_img, 1, cur_locate, 0.5, 0)
            whole_img[i, 0] = cur_locate.transpose(2, 0, 1)[::-1].astype(np.float32) / 255
            for j in range(self.num_parts):
                cur_pred = cv2.applyColorMap(
                    cv2.resize(
                        (tb_pred_map[i, j, :, :, None] * 255).clip(0, 255).astype(np.uint8),
                        (INP_RES, INP_RES)),
                    cv2.COLORMAP_HOT)
                cur_pred = cv2.addWeighted(cur_img, 1, cur_pred, 0.5, 0)
                cv2.putText(cur_pred, datasets.mscoco.PART_LABELS[j], (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1)
                if j in matches_dict:
                    for (pos_x, pos_y), (loc_x, loc_y) in matches_dict[j]:
                        cv2.arrowedLine(cur_pred, (pos_x, pos_y), (loc_x, loc_y), MATCH_COLOR, thickness=1)
                        cv2.circle(cur_pred, (pos_x, pos_y), 3, PRED_COLOR, thickness=-1)
                if len(keypoint_gt[i]) > 0:
                    for kp_person in keypoint_gt[i][:, j]:
                        if kp_person[2] > 0:
                            cv2.circle(cur_pred, (int(round(kp_person[0] * FACTOR)), int(round(kp_person[1] * FACTOR))), 3, GT_COLOR, thickness=-1)
                whole_img[i, j+1] = cur_pred.transpose(2, 0, 1)[::-1].astype(np.float32) / 255

        whole_img = vutils.make_grid(torch.from_numpy(whole_img.reshape((-1, 3, INP_RES, INP_RES))),
                                     nrow=ncols, range=(0, 1), pad_value=0.3)
        config.tb_writer.add_image(title, whole_img, step)


    def summary_histogram(self, n_iter):
        for name, param in self.model.named_parameters():
            config.tb_writer.add_histogram("back." + name, param.clone().cpu().data.numpy(), n_iter, bins="doane")

    @profile
    def process(self, batch, train, detail=None):
        model_hg = self.model
        criterion_mse = self.criterion

        img = batch["img"]
        locate_map = batch["locate_map"]
        keypoint_map_gt = batch["keypoint_map"]
        locate = batch["locate"]
        locate_std = batch["locate_std"]
        locate_in_kp = batch["locate_in_kp"]
        keypoint_gt = batch["keypoint"]
        mask = batch["mask"]

        volatile = not train

        # TODO: volatile and requires_grad
        img_var = torch.autograd.Variable(img.cuda(async=True), volatile=False)
        locate_map_var = torch.autograd.Variable(locate_map.cuda(async=True), requires_grad=True)
        keypoint_map_gt_var = torch.autograd.Variable(keypoint_map_gt.cuda(async=True), volatile=False)
        mask_var = torch.autograd.Variable(mask.cuda(async=True), volatile=False)

        for i in range(len(locate)):
            if locate[i] is None:
                locate[i] = torch.FloatTensor(0)

        for i in range(len(locate_std)):
            if locate_std[i] is not None:
                lsg_i = [float(std) if std is not None else (1.*FACTOR) for std in locate_std[i]]
                locate_std[i] = torch.FloatTensor(lsg_i)

        # Filter keypoints with locate
        # Change visible attribute of outsider to 0
        for bi in range(len(keypoint_gt)):
            lik_bi = locate_in_kp[bi]
            kp_bi = keypoint_gt[bi]
            if lik_bi is not None:
                kp_bi = kp_bi[lik_bi]
                outsider_id = torch.nonzero(((kp_bi < 0) | (kp_bi >= INP_RES))[:, :, :2].sum(dim=-1))
                if len(outsider_id) > 0:
                    kp_bi[outsider_id[:, 0],
                            outsider_id[:, 1],
                            torch.LongTensor([2]).expand(outsider_id.size(0))] = 0
            else:
                kp_bi = torch.FloatTensor(0)
            keypoint_gt[bi] = kp_bi

        keypoint_map_pred_var = model_hg(inp=locate_map_var, merge_inp=img_var, merge=True, free_merge=True)

        loss = criterion_mse(keypoint_map_pred_var, keypoint_map_gt_var, mask_var)

        if detail["summary"]:
            keypoint_pred = self.posemap_parser.parse(keypoint_map_pred_var, mask_var, factor=1)
            matches = list()
            for batch_idx, kp_pred in enumerate(keypoint_pred):
                match_i = list()
                if len(kp_pred) > 0:
                    (pid, pos) = kp_pred
                    for point_idx in range(len(pid)):
                        part_i = pid[point_idx]
                        pos_i = pos[point_idx]
                        if locate_map_var.grad is not None:
                            locate_map_var.grad.zero_()
                        locate_map_grad = torch.autograd.grad(keypoint_map_pred_var[batch_idx, part_i, pos_i[1], pos_i[0]], locate_map_var, retain_graph=True)
                        locate_map_grad = locate_map_grad[0][batch_idx, 0]

                        reason_point = torch.nonzero(locate_map_grad == locate_map_grad.max())[0]
                        match_i.append(((part_i, int(pos_i[0]*4), int(pos_i[1]*4)), (reason_point[1], reason_point[0])))
                matches.append(match_i)

            self.summary_image(img=img,
                               locate_map=locate_map,
                               pred_map=keypoint_map_pred_var.data.cpu(),
                               keypoint_gt=keypoint_gt,
                               matches=matches,
                               title="back/" + ("train" if train else "val"),
                               step=detail["step"])

        if train:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=False)
            self.optimizer.step()

        result= {"loss": loss.clone(), # for freeing graph
                 "acc": 0,
                 "recall": 0,
                 "prec": 0,
                 "index": batch["index"]}
        return result
