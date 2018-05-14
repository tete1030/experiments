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

PAIR = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [5,6],
         [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12],
         [11,13], [13,15], [12,14], [14,16]]

def generate_pair_index():
    pair_indexof = [[] for _ in range(datasets.mscoco.NUM_PARTS)]
    for ip, p in enumerate(PAIR):
        pair_indexof[p[0]].append((ip, 1))
        pair_indexof[p[1]].append((ip, 0))
    return pair_indexof

PAIR_INDEXOF = generate_pair_index()

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
                                             keypoint_label_outsider=True,
                                             keypoint_filter=True)

        self.parser = FieldmapParser(PAIR, PAIR_INDEXOF, detection_thres=self.hparams["eval"]["detection_thres"], group_thres=self.hparams["eval"]["group_thres"], max_num_people=self.hparams["model"]["max_num_people"])

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

        # Make the similarity from 0 to 1
        em_cos = (((em_x_1 * em_x_2 + em_y_1 * em_y_2) / ((em_x_1 ** 2 + em_y_1 ** 2).sqrt() * (em_x_2 ** 2 + em_y_2 ** 2).sqrt()) + 1) / 2) ** 2

        loss_push = (em_cos[push].sum() / self.push_count) if self.push_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(self.current_device), requires_grad=False)
        loss_pull = (1-(em_cos[pull].sum() / self.pull_count)) if self.pull_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(self.current_device), requires_grad=False)

        return loss_push, loss_pull

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
