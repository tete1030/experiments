from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import pose.models as models
from pose.models.parallel import DataParallelModel, DataParallelCriterion, gather
import pose.datasets as datasets
from pose.utils.evaluation import PR_multi
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate
from pose.utils.transforms import fliplr_chwimg, fliplr_map
from pose.models import HeatmapLoss, FieldmapLoss
from pose.utils.group import FieldmapParser
from pose.utils.imutils import HeatmapGenerator
import munkres
from munkres import Munkres

import sys

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np
import torchvision.utils as vutils

import unittest
import threading

try:
    profile
except NameError:
    profile = lambda func: func

INP_RES = 512
FACTOR = 4
OUT_RES = INP_RES // FACTOR
INP_EXTRA_RES = [INP_RES//2, INP_RES*2]
INP_EVAL_RES = [INP_RES] + INP_EXTRA_RES

# PAIR = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [5,6],
#          [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12],
#          [11,13], [13,15], [12,14], [14,16]]

PAIR = [[17, 6], [17, 5], [6, 8], [8, 10], [5, 7], [7, 9],
        [17, 12], [12, 14], [14, 16], [17, 11], [11, 13],
        [13, 15], [17, 0], [0, 2], [2, 4], [0, 1], [1, 3],
        [6, 4], [5, 3]]

class Experiment(object):
    def __init__(self, hparams):
        self.num_parts = datasets.mscoco.NUM_PARTS
        self.hparams = hparams
        self.model = DataParallelModel(
            models.PoseHGNet(
                inp_dim=3,
                out_dim=self.num_parts + len(PAIR)*4,
                nstack=hparams["model"]["nstack"],
                hg_dim=hparams["model"]["hg_dim"],
                increase=hparams["model"]["increase"],
                bn=hparams["model"]["bn"]).cuda())

        ae2dloss = DataParallelCriterion(AE2DLoss(pair=PAIR).cuda(), comp_mean=True, store_replicas=True)
        for i, did in enumerate(ae2dloss.device_ids):
            ae2dloss.replicas[i].current_device = did

        self.criterion = torch.nn.ModuleList([
            DataParallelCriterion(HeatmapLoss().cuda(), comp_mean=True, store_replicas=True),
            ae2dloss])

        self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                          lr=hparams['learning_rate'],
                                          weight_decay=hparams['weight_decay'])

        self.coco = COCO("data/mscoco/person_keypoints_train2014.json")

        def keypoint_extender(keypoint):
            keypoint = np.concatenate((keypoint, np.zeros((keypoint.shape[0], 1, 3), dtype=np.float32)), axis=1)
            keypoint[:, -1, :2] = (keypoint[:, 5, :2] + keypoint[:, 6, :2]) / 2
            keypoint[:, -1, 2] = ((keypoint[:, 5, 2] > 0) & (keypoint[:, 6, 2] > 0)).astype(np.float32)
            return keypoint

        self.train_dataset = datasets.COCOPose("data/mscoco/images",
                                               self.coco,
                                               "data/mscoco/split.pth",
                                               "data/mscoco/mean_std.pth",
                                               train=True,
                                               single_person=False,
                                               img_res=[INP_RES],
                                               return_img_transform=True,
                                               mask_res=OUT_RES,
                                               kpmap_res=OUT_RES,
                                               kpmap_select="all",
                                               keypoint_res=OUT_RES,
                                               keypoint_extender=keypoint_extender,
                                               keypoint_label_outsider=True,
                                               keypoint_filter=True)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             img_res=INP_EVAL_RES,
                                             return_img_transform=True,
                                             mask_res=OUT_RES,
                                             kpmap_res=OUT_RES,
                                             kpmap_select="all",
                                             keypoint_res=OUT_RES,
                                             keypoint_extender=keypoint_extender,
                                             keypoint_label_outsider=True,
                                             keypoint_filter=True)

        self.parser = AE2DParser(PAIR, detection_thres=self.hparams["eval"]["detection_thres"], group_thres=self.hparams["eval"]["group_thres"], max_num_people=self.hparams["model"]["max_num_people"])

        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

    def epoch(self, epoch):
        self.hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, self.hparams['learning_rate'], self.hparams['schedule'], self.hparams['lr_gamma'])

    # def summary_image(self, img, pred, gt, title, step):
    #     tb_num = 3
    #     tb_img = img[:tb_num].numpy() + self.train_dataset.mean[None, :, None, None]
    #     tb_gt = gt[:tb_num].numpy()
    #     tb_pred = pred[:tb_num].numpy()
    #     show_img = np.zeros((tb_num * 2, 3, INP_RES, INP_RES))
    #     for iimg in range(tb_num):
    #         cur_img = (tb_img[iimg][::-1].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    #         cur_gt = cv2.applyColorMap(
    #             cv2.resize(
    #                 (tb_gt[iimg].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
    #                 (INP_RES, INP_RES)),
    #             cv2.COLORMAP_HOT)
    #         cur_gt = cv2.addWeighted(cur_img, 1, cur_gt, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
    #         show_img[iimg] = cur_gt
    #         cur_pred = cv2.applyColorMap(
    #             cv2.resize(
    #                 (tb_pred[iimg].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
    #                 (INP_RES, INP_RES)),
    #             cv2.COLORMAP_HOT)
    #         cur_pred = cv2.addWeighted(cur_img, 1, cur_pred, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
    #         show_img[tb_num + iimg] = cur_pred

    #     show_img = vutils.make_grid(torch.from_numpy(show_img), nrow=tb_num, range=(0, 1))
    #     config.tb_writer.add_image(title, show_img, step)

    def summary_histogram(self, n_iter):
        for name, param in self.model.named_parameters():
            config.tb_writer.add_histogram(config.exp_name + "." + name, param.clone().cpu().data.numpy(), n_iter, bins="doane")

    def evaluate(self, image_ids, ans):
        if len(ans) > 0:
            coco_dets = self.coco.loadRes(ans)
            coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
            coco_eval.params.imgIds = list(image_ids)
            coco_eval.params.catIds = [1]
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            print("No points")

    @profile
    def process(self, batch, train, detail=None):
        def extract_map(output_var):
            assert output_var[-1].size(1) == self.num_parts + len(PAIR)*4
            det_map = output_var[-1].data[:, :self.num_parts].cpu()
            field_map = output_var[-1].data[:, self.num_parts:].cpu()
            return det_map, field_map

        heatmaploss = self.criterion[0]
        ae2dloss = self.criterion[1]

        imgs = batch["img"]
        det_map = batch["keypoint_map"]
        det_mask = batch["mask"]
        transform_mats = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        keypoint = batch["keypoint"]
        volatile = not train

        split_size = (len(keypoint) + len(ae2dloss.replicas) - 1) // len(ae2dloss.replicas)
        keypoint_scattered = split_data(keypoint, split_size)
        ae2dloss.replicas_each("prepare", ind_args=map(lambda a: (a,), keypoint_scattered), shared_kwargs=dict(_async=False))

        det_map_var = Variable(det_map, volatile=volatile)
        det_map_mask_var = Variable(det_mask, volatile=volatile)

        output_var = zip(*self.model(Variable(imgs[0], volatile=volatile)))
        heatmap_var = map(lambda out_stack: map(lambda out_gpu: out_gpu[:, :self.num_parts], out_stack), output_var)
        emmap_var = map(lambda out_stack: map(lambda out_gpu: out_gpu[:, self.num_parts:], out_stack), output_var)

        kplosses = []
        emlosses_push = []
        emlosses_pull = []
        for istack in range(self.hparams["model"]["nstack"]):
            kplosses.append(heatmaploss(heatmap_var[istack], det_map_var, det_map_mask_var))
            eml_push, eml_pull = ae2dloss(emmap_var[istack])
            emlosses_push.append(eml_push)
            emlosses_pull.append(eml_pull)

        kploss = sum(kplosses) / len(kplosses)
        emloss_push = sum(emlosses_push) / len(emlosses_push)
        emloss_pull = sum(emlosses_pull) / len(emlosses_pull)

        loss = kploss * self.hparams["model"]["loss_det_cof"] + emloss_push * self.hparams["model"]["loss_em_push_cof"] + emloss_pull * self.hparams["model"]["loss_em_pull_cof"]

        if (loss.data != loss.data).any():
            import pdb; pdb.set_trace()
            # raise RuntimeError("loss is nan")


        phase_str = "train" if train else "valid"
        config.tb_writer.add_scalars(config.exp_name + "/loss_det", {phase_str: kploss.data.cpu()[0]}, detail["step"])
        config.tb_writer.add_scalars(config.exp_name + "/loss_em_push", {phase_str: emloss_push.data.cpu()[0]}, detail["step"])
        config.tb_writer.add_scalars(config.exp_name + "/loss_em_pull", {phase_str: emloss_pull.data.cpu()[0]}, detail["step"])

        result = {
            "loss": loss,
            "acc": 0,
            "recall": 0,
            "prec": None,
            "index": batch["index"],
            "pred": None
        }
    
        return result

class AE2DLoss(torch.nn.Module):
    def __init__(self, pair):
        super(AE2DLoss, self).__init__()
        self.pair = pair
        self.thread = None
        self.iS = None
        self.iC = None
        self.X_1 = None
        self.Y_1 = None
        self.X_2 = None
        self.Y_2 = None
        self.pull = None
        self.push = None
        self.pull_count = None
        self.push_count = None
        self.current_device = None

    def prepare(self, keypoint, _async=True):
        def _worker(self, keypoint):
            pull = []
            iS = []
            iC = []
            Y_1 = []
            X_1 = []
            Y_2 = []
            X_2 = []
            for isample in range(len(keypoint)):
                kp_samp = keypoint[isample]
                if kp_samp is None:
                    continue
                for ipair in range(len(self.pair)):
                    pr = self.pair[ipair]
                    pr_1 = pr[0]
                    pr_2 = pr[1]
                    sel_1_mask = (kp_samp[:, pr_1, 2] > 0)
                    sel_2_mask = (kp_samp[:, pr_2, 2] > 0)
                    sel_1_ids = sel_1_mask.nonzero()
                    sel_2_ids = sel_2_mask.nonzero()
                    if len(sel_1_ids) == 0 or len(sel_2_ids) == 0:
                        continue
                    sel_1_ids = sel_1_ids[:, 0]
                    sel_2_ids = sel_2_ids[:, 0]
                    # sel_pull_mask = (sel_1_mask & sel_2_mask)
                    # sel_pull_ids = sel_pull_mask.nonzero()
                    # if len(sel_pull_ids) > 0:
                    #     sel_pull_ids = sel_pull_ids[:, 0]
                    #     iS_pull.append(torch.LongTensor(sel_pull_ids.size(0)).fill_(isample))
                    #     iC_pull.append(torch.LongTensor(sel_pull_ids.size(0)).fill_(ipair*2))
                    #     X_1_pull.append(kp_samp[sel_pull_ids][:, pr_1, 0])
                    #     Y_1_pull.append(kp_samp[sel_pull_ids][:, pr_1, 1])
                    #     X_2_pull.append(kp_samp[sel_pull_ids][:, pr_2, 0])
                    #     Y_2_pull.append(kp_samp[sel_pull_ids][:, pr_2, 1])

                    sel_comb_1, sel_comb_2 = np.meshgrid(sel_1_ids.numpy(), sel_2_ids.numpy(), indexing="ij")
                    sel_comb_1 = torch.from_numpy(sel_comb_1.ravel()).long()
                    sel_comb_2 = torch.from_numpy(sel_comb_2.ravel()).long()
                    pull.append((sel_comb_1 == sel_comb_2))
                    iS.append(torch.LongTensor(sel_comb_1.size(0)).fill_(isample))
                    iC.append(torch.LongTensor(sel_comb_1.size(0)).fill_(ipair*2))
                    X_1.append(kp_samp[sel_comb_1][:, pr_1, 0])
                    Y_1.append(kp_samp[sel_comb_1][:, pr_1, 1])
                    X_2.append(kp_samp[sel_comb_2][:, pr_2, 0])
                    Y_2.append(kp_samp[sel_comb_2][:, pr_2, 1])

            if len(iS) > 0:
                with torch.cuda.device(self.current_device):
                    self.iS = torch.cat(iS).cuda(async=True)
                    self.iC = torch.cat(iC).cuda(async=True)
                    self.X_1 = torch.cat(X_1).long().cuda(async=True)
                    self.Y_1 = torch.cat(Y_1).long().cuda(async=True)
                    self.X_2 = torch.cat(X_2).long().cuda(async=True)
                    self.Y_2 = torch.cat(Y_2).long().cuda(async=True)
                    pull = torch.cat(pull)
                    push = (~pull)
                    self.pull = pull.cuda(async=True)
                    self.push = push.cuda(async=True)
                    self.pull_count = pull.int().sum()
                    self.push_count = push.int().sum()
            else:
                self.pull_count = 0
                self.push_count = 0

        self.iS = None
        self.iC = None
        self.X_1 = None
        self.Y_1 = None
        self.X_2 = None
        self.Y_2 = None
        self.pull = None
        self.push = None
        self.pull_count = None
        self.push_count = None

        if _async:
            self.thread = threading.Thread(target=_worker, args=(self, keypoint))
            self.thread.start()
        else:
            _worker(self, keypoint)

    def forward(self, pred_field):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

        if self.pull_count > 0 or self.push_count > 0:
            assert self.iS is not None

            iS = self.iS
            iC = self.iC
            X_1 = self.X_1
            Y_1 = self.Y_1
            X_2 = self.X_2
            Y_2 = self.Y_2
            pull = self.pull
            push = self.push

            em_x_1 = pred_field[iS, iC, Y_1, X_1]
            em_y_1 = pred_field[iS, iC+1, Y_1, X_1]

            em_x_2 = pred_field[iS, iC, Y_2, X_2]
            em_y_2 = pred_field[iS, iC+1, Y_2, X_2]

            # TODO: situation: magnitude too small
            # Make the similarity from 0 to 1
            em_cos = (((em_x_1 * em_x_2 + em_y_1 * em_y_2) / ((em_x_1 ** 2 + em_y_1 ** 2).sqrt() * (em_x_2 ** 2 + em_y_2 ** 2).sqrt()) + 1) / 2) ** 2

        loss_push = (em_cos[push].sum() / self.push_count) if self.push_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(self.current_device), requires_grad=False)
        loss_pull = (1-(em_cos[pull].sum() / self.pull_count)) if self.pull_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(self.current_device), requires_grad=False)

        return loss_push, loss_pull

class AE2DParser(object):
    def __init__(self, pair, detection_thres=0.1, group_thres=0.1, max_num_people=30):
        self.pair = pair
        self.detection_thres = detection_thres
        self.group_thres = group_thres
        self.max_num_people = max_num_people
        self.pool = torch.nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        # suppose det is a tensor
        det = torch.autograd.Variable(det, volatile=True)
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det.data

    def parse(self, det, field):
        assert isinstance(det, torch.FloatTensor) and isinstance(field, torch.FloatTensor)
        num_samp = det.size(0)
        num_joint = det.size(1)
        height = det.size(2)
        width = det.size(3)
        num_channel = len(self.pair) * 2
        det = self.nms(det)

        topkval, topkind = det.view(det.size()[:2] + (-1,)).topk(self.max_num_people, dim=-1)

        det = det.numpy()
        field = field.numpy()
        topkval = topkval.numpy()
        topkind = topkind.numpy().astype(np.int32)
        # topkloc = np.stack([topkind % width, topkind / width], axis=-1)

        mask_topk_thr = (topkval > self.detection_thres)
        field = field.reshape(field.shape[:2] + (-1,))

        preds = []
        for isamp in range(num_samp):
            topkind_samp_thr = dict()
            ijoint_idet_2_iperson = dict()
            iperson_ijoint_2_idet = list()
            person_counter = 0
            
            for ipair, pr in enumerate(self.pair):
                ijoint1 = pr[0]
                ijoint2 = pr[1]

                if ijoint1 not in topkind_samp_thr:
                    sel_topk_joint1_thr = mask_topk_thr[isamp, ijoint1].nonzero()[0]
                    topkind_samp_thr[ijoint1] = topkind[isamp, ijoint1][sel_topk_joint1_thr]

                if ijoint2 not in topkind_samp_thr:
                    sel_topk_joint2_thr = mask_topk_thr[isamp, ijoint2].nonzero()[0]
                    topkind_samp_thr[ijoint2] = topkind[isamp, ijoint2][sel_topk_joint2_thr]

                if ijoint1 not in ijoint_idet_2_iperson:
                    ijoint_idet_2_iperson[ijoint1] = [-1] * topkind_samp_thr[ijoint1].shape[0]

                if ijoint2 not in ijoint_idet_2_iperson:
                    ijoint_idet_2_iperson[ijoint2] = [-1] * topkind_samp_thr[ijoint2].shape[0]

                topkind_joint1 = topkind_samp_thr[ijoint1]
                topkind_joint2 = topkind_samp_thr[ijoint2]

                if topkind_joint1.shape[0] == 0 or topkind_joint2.shape[0] == 0:
                    continue

                x1 = field[isamp, ipair*2, topkind_joint1][:, np.newaxis]
                y1 = field[isamp, ipair*2+1, topkind_joint1][:, np.newaxis]

                x2 = field[isamp, ipair*2, topkind_joint2][np.newaxis, :]
                y2 = field[isamp, ipair*2+1, topkind_joint2][np.newaxis, :]

                # TODO: situation: magnitude too small
                sim_neg = ((-(x1 * x2 + y1 * y2) / np.sqrt((x1 ** 2 + y1 ** 2) * (x2 ** 2 + y2 ** 2)) + 1) / 2)

                # mk = Munkres()

                # sim_neg[sim_neg > self.group_thres] = 1e10
                # sim_neg_pad = mk.pad_matrix(sim_neg.tolist(), 1e10)
                
                # # \T\O\D\O: mark conflict person as DISALLOWED ?

                # pairs_det = mk.compute(sim_neg_pad)

                pairs_det = []
                sim_neg_clone = sim_neg.copy()
                while True:
                    sim_neg_min_ind = sim_neg_clone.argmin()
                    idet1 = sim_neg_min_ind / sim_neg_clone.shape[1]
                    idet2 = sim_neg_min_ind % sim_neg_clone.shape[1]
                    if sim_neg_clone[idet1, idet2] > self.group_thres:
                        break
                    pairs_det.append((idet1, idet2))
                    sim_neg_clone[idet1] = 1e10
                    sim_neg_clone[:, idet2] = 1e10

                for idet1, idet2 in pairs_det:
                    # if idet1 >= sim_neg.shape[0] or idet2 >= sim_neg.shape[1] or sim_neg[idet1, idet2] > self.group_thres:
                    #     continue
                    iperson1 = ijoint_idet_2_iperson[ijoint1][idet1]
                    iperson2 = ijoint_idet_2_iperson[ijoint2][idet2]
                    if iperson1 != -1 and iperson2 == -1:
                        # assigning joint2
                        ijoint_idet_2_iperson[ijoint2][idet2] = iperson1
                        iperson_ijoint_2_idet[iperson1][ijoint2] = idet2
                    elif iperson1 == -1 and iperson2 != -1:
                        # assigning joint1
                        ijoint_idet_2_iperson[ijoint1][idet1] = iperson2
                        iperson_ijoint_2_idet[iperson2][ijoint1] = idet1
                    elif iperson1 == -1 and iperson2 == -1:
                        # assigning both
                        ijoint_idet_2_iperson[ijoint1][idet1] = person_counter
                        ijoint_idet_2_iperson[ijoint2][idet2] = person_counter
                        iperson_ijoint_2_idet.append([-1] * num_joint)
                        assert len(iperson_ijoint_2_idet) == person_counter + 1
                        iperson_ijoint_2_idet[person_counter][ijoint1] = idet1
                        iperson_ijoint_2_idet[person_counter][ijoint2] = idet2
                        person_counter += 1
                    elif iperson1 != iperson2:
                        # combine
                        person1 = np.array(iperson_ijoint_2_idet[iperson1], dtype=np.int32)
                        person2 = np.array(iperson_ijoint_2_idet[iperson2], dtype=np.int32)
                        mask_person2 = (person2 >= 0)
                        if ((person1 >= 0) & mask_person2).any():
                            # TODO: conflict
                            pass
                        else:
                            person1[mask_person2] = person2[mask_person2]
                            iperson_ijoint_2_idet[iperson1] = person1.tolist()
                            iperson_ijoint_2_idet[iperson2] = [-1] * num_joint
                            for _ijoint_person2, _idet_person2 in enumerate(person2):
                                if _idet_person2 >= 0:
                                    ijoint_idet_2_iperson[_ijoint_person2][_idet_person2] = iperson1
            
            iperson_ijoint_2_idet = np.array(iperson_ijoint_2_idet, dtype=np.int32)
            sel_person = ((iperson_ijoint_2_idet >= 0).astype(np.int32).sum(axis=1) > 0).nonzero()[0]
            iperson_ijoint_2_idet = iperson_ijoint_2_idet[sel_person]

            pred = np.zeros(iperson_ijoint_2_idet.shape + (3,), dtype=np.int32)

            if sel_person.shape[0] > 0:
                for ijoint in range(num_joint):
                    idet_2_ind = topkind_samp_thr[ijoint]
                    if idet_2_ind.shape[0] == 0:
                        continue
                    sel_valid_joint_person = (iperson_ijoint_2_idet[:, ijoint] >= 0).nonzero()[0]
                    ind = idet_2_ind[iperson_ijoint_2_idet[sel_valid_joint_person, ijoint]]

                    pred[sel_valid_joint_person, ijoint, 0] = ind % width
                    pred[sel_valid_joint_person, ijoint, 1] = ind / width
                    pred[sel_valid_joint_person, ijoint, 2] = 1

            preds.append(pred.astype(np.float32))

        return preds

# Input: 
#   det: num_parts(17) x h x w
#   tag: num_parts(17) x h x w x (num_scale_near_one*2)
#   kp:  num_parts(17) x (3+num_scale_near_one*2)
# Return: 
def refine(det, tag, keypoints):
    """
    Given initial keypoint predictions, we identify missing joints
    """
    if len(tag.shape) == 3:
        tag = tag[:,:,:,None]

    # tags: num_parts(17) x [(num_scale_near_one*2)]
    tags = []
    for i in range(keypoints.shape[0]):
        # keypoint detected
        if keypoints[i, 2] > 0:
            y, x = keypoints[i][:2].astype(np.int32)
            tags.append(tag[i, x, y])

    # averaged detected tag of the person
    # prev_tag: (num_scale_near_one*2)
    prev_tag = np.mean(tags, axis = 0)
    ans = []

    for i in range(keypoints.shape[0]):
        # tmp: h x w
        tmp = det[i, :, :]
        # tt: h x w
        tt = (((tag[i, :, :] - prev_tag[None, None, :])**2).sum(axis = 2)**0.5 )
        tmp2 = tmp - np.round(tt)

        # !!relax the prerequisition to max point of a feature map (better bad than not)
        x, y = np.unravel_index( np.argmax(tmp2), tmp.shape )
        xx = x
        yy = y
        val = tmp[x, y]
        x += 0.5
        y += 0.5

        if tmp[xx, min(yy+1, det.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, det.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25

        x, y = np.array([y,x])
        ans.append((x, y, val))
    ans = np.array(ans)

    if ans is not None:
        for i in range(17):
            if ans[i, 2]>0 and keypoints[i, 2]==0:
                keypoints[i, :2] = ans[i, :2]
                keypoints[i, 2] = 1 

    return keypoints

def batch_resize(im, res):
    im_pre_shape = im.shape[:-2]
    im_post_shape = im.shape[-2:]
    if im_post_shape == res:
        return im
    im = im.reshape((-1,) + im_post_shape)
    return np.array([cv2.resize(im[i], res) for i in range(im.shape[0])]).reshape(im_pre_shape + res)

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)

def generate_ans(image_ids, preds, scores):
    ans = []
    for sample_i in range(len(preds)):
        image_id = image_ids[sample_i]
        for person_i in range(len(preds[sample_i])):
            # val: num_parts(17) x 3
            val = preds[sample_i][person_i]
            score = scores[sample_i][person_i]
            if val[:, 2].max()>0:
                tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(score)}
                # p: average detected locations
                p = val[val[:, 2] > 0][:, :2].mean(axis = 0)
                for j in val:
                    if j[2]>0.:
                        tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
                    else:
                        # TRICK: for not detected points, place them at the average point
                        tmp["keypoints"] += [float(p[0]), float(p[1]), 1]
                ans.append(tmp)
    return ans

def split_data(data, split_size):
    """Splits the data into equally sized chunks (if possible).

    Last chunk will be smaller if the data size along a given dimension
    is not divisible by ``split_size``.

    Arguments:
        data (list): data to split.
        split_size (int): size of a single chunk.
    """
    size = len(data)
    num_splits = (size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - size)

    def get_split_size(i):
        return split_size if i < num_splits - 1 else last_split_size
    return tuple([data[i*split_size : i*split_size+get_split_size(i)] for i in range(0, num_splits)])


class TestOutput(object):
    def __init__(self):
        hparams = dict(
            learning_rate=2e-4,
            weight_decay=0,
            model=dict(
                nstack=4,
                hg_dim=256,
                increase=128,
                bn=False,
                max_num_people=30,
                loss_det_cof=1,
                loss_em_push_cof=1e-4,
                loss_em_pull_cof=1e-3),
            eval=dict(
                detection_thres=0.05,
                group_thres=0.1)
        )
        self.exp = Experiment(hparams)
        self.dataloader = torch.utils.data.DataLoader(self.exp.val_dataset,
                                                      batch_size=8,
                                                      shuffle=False,
                                                      num_workers=5,
                                                      collate_fn=self.exp.test_collate_fn,
                                                      pin_memory=True)
        self.data = next(iter(self.dataloader))
        # self.pair = [[-1, 6], [-1, 5], [6, 8], [8, 10], [5, 7], [7, 9], [-1, 12], [12, 14], [14, 16], [-1, 11], [11, 13], [13, 15], [-1, 0], [0, 2], [2, 4], [0, 1], [1, 3], [6, 4], [5, 3]]
        # 17 is mean of 6 and 7
        self.pair = [[17, 6], [17, 5], [6, 8], [8, 10], [5, 7], [7, 9], [17, 12], [12, 14], [14, 16], [17, 11], [11, 13], [13, 15], [17, 0], [0, 2], [2, 4], [0, 1], [1, 3], [6, 4], [5, 3]]
        print("Data loaded")

    def generate_output(self, transform_back=True):
        image_ids = list(self.data["img_index"])
        det_map = self.data["keypoint_map"]
        det_mask = self.data["mask"]
        transform_mats = self.data["img_transform"]
        img_flipped = self.data["img_flipped"]
        keypoint = self.data["keypoint"]

        field_map = np.zeros((det_map.size(0), len(PAIR)*2, det_map.size(2), det_map.size(3)), dtype=np.float32)
        for isamp in range(len(keypoint)):
            keypoint_samp = keypoint[isamp]
            if keypoint_samp is None:
                continue
            keypoint_samp = keypoint_samp.numpy()
            num_person = len(keypoint_samp)
            
            for ipair, pr in enumerate(PAIR):
                ijoint1 = pr[0]
                ijoint2 = pr[1]
                joint1 = keypoint_samp[:, ijoint1]
                joint2 = keypoint_samp[:, ijoint2]
                sel_person_valid_union = ((joint1[:, 2] > 0) | (joint2[:, 2] > 0)).nonzero()[0]
                num_person_union = sel_person_valid_union.shape[0]

                if num_person_union == 0:
                    continue

                interval = 2 * np.pi / float(num_person_union)
                phase = np.random.rand(1)[0] * 2 * np.pi
                agl = np.arange(num_person_union, dtype=np.float32) * interval + np.random.randn(num_person_union) * interval * 0.001 / 2
                # agl = agl + phase
                np.random.shuffle(agl)
                for iagl, iperson in enumerate(sel_person_valid_union):
                    if joint1[iperson, 2] > 0:
                        rand_mag = np.random.rand(1)[0]
                        rand_mag = 1
                        self.exp.val_dataset.heatmap_gen(joint1[iperson, :2], ipair*2, field_map[isamp], normalize_factor=np.cos(agl[iagl]) * rand_mag, mingle_mode="add")
                        self.exp.val_dataset.heatmap_gen(joint1[iperson, :2], ipair*2+1, field_map[isamp], normalize_factor=np.sin(agl[iagl]) * rand_mag, mingle_mode="add")

                    if joint2[iperson, 2] > 0:
                        rand_mag = np.random.rand(1)[0]
                        rand_mag = 1
                        self.exp.val_dataset.heatmap_gen(joint2[iperson, :2], ipair*2, field_map[isamp], normalize_factor=np.cos(agl[iagl]) * rand_mag, mingle_mode="add")
                        self.exp.val_dataset.heatmap_gen(joint2[iperson, :2], ipair*2+1, field_map[isamp], normalize_factor=np.sin(agl[iagl]) * rand_mag, mingle_mode="add")

        field_map = torch.from_numpy(field_map).float()

        preds = self.exp.parser.parse(det_map, field_map)
        preds_new = []
        scores = []
        for batch_i, sample_pred in enumerate(preds):
            if transform_back:
                inverse_mat = np.linalg.pinv(transform_mats[batch_i][0])[:2]
                sample_pred[:, :, :2] = kpt_affine(sample_pred[:, :, :2], inverse_mat)
            preds_new.append(sample_pred)
            scores.append(sample_pred[:, :, 2].mean(axis=1))
        return {
            "pred": preds_new,
            "score": scores,
            "image_id": image_ids,
            "det_map": det_map,
            "field_map": field_map,
            "keypoint": keypoint,
            "img": self.data["img"]
        }

    def plot(self):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import cv2
        def draw_pose(img, pose, factor=1):
            if torch.is_tensor(pose):
                pose = pose.numpy()

            for iperson, personjoint in enumerate(pose):
                color = np.random.randint(0, 256, size=3, dtype=np.uint8).tolist()

                personjoint = personjoint.copy()
                personjoint[:, :2] = personjoint[:, :2] * factor
                personjoint = personjoint.astype(int)

                for ijoint, joint in enumerate(personjoint):
                    if joint[2] > 0:
                        cv2.circle(img, tuple(joint[:2].tolist()), 3, color, thickness=-1)
                for iconnect, connect in enumerate(self.pair):
                    joint1 = personjoint[connect[0]]
                    joint2 = personjoint[connect[1]]
                    if joint1[2] > 0 and joint2[2] > 0:
                        cv2.line(img, tuple(joint1[:2].tolist()), tuple(joint2[:2].tolist()), color, thickness=1)

        def generate_field_img(field_x, field_y):
            field_rho = np.sqrt(field_x**2 + field_y**2)
            field_phi = np.arctan2(field_y, field_x)
            fieldimg_h = (field_phi/np.pi + 1) / 2
            assert ((fieldimg_h >= 0) & (fieldimg_h <= 1)).all()
            fieldimg_s = np.ones(fieldimg_h.shape)
            fieldimg_v = field_rho
            fieldimg = mpl.colors.hsv_to_rgb(np.stack((fieldimg_h, fieldimg_s, fieldimg_v), axis=-1))
            return fieldimg

        result = self.generate_output(transform_back=False)
        preds = result["pred"]
        scores = result["score"]
        image_ids = result["image_id"]
        gts = result["keypoint"]
        field_map = result["field_map"].numpy()
        img = self.exp.val_dataset.restore_image(result["img"][0])
        # gt, pred
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20), gridspec_kw={"wspace": 0, "hspace": 0, "left": 0, "top": 1, "right": 1, "bottom": 0})
        num_pair = len(PAIR)
        num_col = 6
        num_row = (num_pair + num_col - 1) // num_col
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            fig = plt.figure(figsize=(num_col * 7, num_col * 7 // 2 + num_row * 7))
            gs = gridspec.GridSpec(2, 1, wspace=0, hspace=0, left=0, top=1, right=1, bottom=0, height_ratios=[num_col / 2., num_row])
            gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0, hspace=0, subplot_spec=gs[0])
            gs1 = gridspec.GridSpecFromSubplotSpec(num_row, num_col, wspace=0.05, hspace=0, subplot_spec=gs[1])

            gt = gt.numpy()
            gt = np.concatenate((gt, np.zeros((gt.shape[0], 1, 3), dtype=np.float32)), axis=1)
            gt[:, 17, :2] = (gt[:, 5, :2] +  gt[:, 6, :2]) / 2
            gt[:, 17, 2] = ((gt[:, 5, 2] > 0) & (gt[:, 6, 2] > 0)).astype(np.float32)
            pred = np.concatenate((pred, np.zeros((pred.shape[0], 1, 3), dtype=np.float32)), axis=1)
            pred[:, 17, :2] = (pred[:, 5, :2] + pred[:, 6, :2]) / 2
            pred[:, 17, 2] = ((pred[:, 5, 2] > 0) & (pred[:, 6, 2] > 0)).astype(np.float32)

            draw_img_gt = img[i].copy()
            draw_img_pred = draw_img_gt.copy()
            draw_pose(draw_img_gt, gt, 4)
            draw_pose(draw_img_pred, pred, 4)
            ax = fig.add_subplot(gs0[0])
            ax.imshow(draw_img_gt)
            ax.axis("off")
            ax = fig.add_subplot(gs0[1])
            ax.imshow(draw_img_pred)
            ax.axis("off")

            fm = batch_resize(field_map[i], draw_img_gt.shape[-3:-1])
            for ipair in range(len(PAIR)):
                img_field = (generate_field_img(fm[ipair*2], fm[ipair*2+1]) * 255).round().clip(0, 255).astype(np.uint8)
                draw_img_field = cv2.addWeighted(img[i], 1, img_field, 0.8, 0)
                for joint_person in gt:
                    for ipairjoint in range(2):
                        joint = joint_person[PAIR[ipair][ipairjoint]]
                        if joint[2] > 0:
                            pt = tuple((joint[:2] * 4).astype(np.int32).tolist())
                            force = (int(fm[ipair*2, pt[1], pt[0]]*20), int(fm[ipair*2+1, pt[1], pt[0]]*20))
                            cv2.arrowedLine(draw_img_field, pt, (pt[0]+force[0], pt[1]+force[1]), (255, 255, 255), thickness=2)

                ax = fig.add_subplot(gs1[ipair // num_col, ipair % num_col])
                ax.imshow(draw_img_field)
                ax.axis("off")
            plt.show()

class UnitTest(unittest.TestCase):
    def setUp(self):
        self.to = TestOutput()

    def run(self, result=None):
        def __add_error_replacement(_, err):
            etype, evalue, etraceback = err
            raise etype, evalue, etraceback

        if result is not None:
            result.addError = __add_error_replacement
            result.addFailure = __add_error_replacement
        super(Test, self).run(result)

    def test_evaluate(self):
        result = self.to.generate_output()
        preds = result["pred"]
        scores = result["score"]
        image_ids = result["image_id"]
        ans = generate_ans(image_ids, preds, scores)
        self.to.exp.evaluate(image_ids, ans)

if __name__ == "__main__":
    # unittest.main()
    TestOutput().plot()
