from __future__ import print_function, absolute_import
import torch
from torch.autograd import Variable
import pose.models as models
from pose.models.parallel import DataParallelModel, DataParallelCriterion, gather
from torch.nn import DataParallel
import pose.datasets as datasets
from pose.utils.evaluation import PR_multi
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate
from pose.utils.transforms import fliplr_chwimg, fliplr_map, fliplr_pts
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

import threading

try:
    profile
except NameError:
    profile = lambda func: func

FACTOR = 4

# PAIR = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [5,6],
#          [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12],
#          [11,13], [13,15], [12,14], [14,16]]

PAIR = [[17, 6], [17, 5], [6, 8], [8, 10], [5, 7], [7, 9],
        [17, 12], [12, 14], [14, 16], [17, 11], [11, 13],
        [13, 15], [17, 0], [0, 2], [2, 4], [0, 1], [1, 3],
        [6, 4], [5, 3]]

NUM_PARTS = datasets.mscoco.NUM_PARTS + 1

def compute_nearest_distance(pair):
    num_joint = NUM_PARTS
    pair_indexof = [[] for _ in range(num_joint)]
    for ip, p in enumerate(pair):
        pair_indexof[p[0]].append((ip, 1))
        pair_indexof[p[1]].append((ip, 0))

    def walk(ijoint_start, ijoint_dest, track=[0] * num_joint, limit=num_joint, step=1):
        if track[ijoint_start] > 0:
            return -1
        
        if limit <= 0:
            return -1

        track[ijoint_start] = step
        minimum = limit
        counter = 0
        for comp_pair, comp_i in pair_indexof[ijoint_start]:
            ijoint_next = pair[comp_pair][comp_i]
            if ijoint_next == ijoint_dest:
                counter += 1
                if 1 < minimum:
                    minimum = 1
                break
            else:
                ret = walk(ijoint_next, ijoint_dest, track, minimum-1, step+1)
                if ret != -1:
                    counter += 1
                    if ret + 1 < minimum:
                        minimum = ret + 1
        track[ijoint_start] = 0
        if counter > 0:
            return minimum
        else:
            return -1

    dismat = np.zeros((num_joint, num_joint), dtype=np.int32)
    for ijoint1 in range(num_joint):
        dismat[ijoint1, ijoint1] = 0
        if ijoint1+1 < num_joint:
            for ijoint2 in range(ijoint1+1, num_joint):
                ret = walk(ijoint1, ijoint2)
                assert 0 < ret < num_joint
                dismat[ijoint1, ijoint2] = ret
                dismat[ijoint2, ijoint1] = ret

    # Release cycle reference in nested recurring function (python2)
    walk = None

    return dismat

DISTANCE_MAT = compute_nearest_distance(PAIR)

class Experiment(object):
    def __init__(self, hparams):
        self.num_parts = datasets.mscoco.NUM_PARTS + 1
        self.hparams = hparams
        self.model = DataParallel(
            models.PoseHGNet(
                inp_dim=3,
                out_dim=self.num_parts + len(PAIR)*4,
                nstack=hparams["model"]["nstack"],
                hg_dim=hparams["model"]["hg_dim"],
                increase=hparams["model"]["increase"],
                bn=hparams["model"]["bn"]).cuda())

        ae2dloss = AE2DLoss(pair=PAIR).cuda()
        ae2dloss.current_device = torch.cuda.device_count()-1

        self.criterion = torch.nn.ModuleList([
            HeatmapLoss().cuda(),
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
                                               img_res=[self.hparams["model"]["inp_res"]],
                                               mask_res=self.hparams["model"]["out_res"],
                                               kpmap_res=self.hparams["model"]["out_res"],
                                               kpmap_select="all_ex",
                                               keypoint_res=self.hparams["model"]["out_res"],
                                               keypoint_extender=keypoint_extender,
                                               keypoint_label_outsider=True,
                                               keypoint_filter=True)

        self.val_dataset = datasets.COCOPose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/split.pth",
                                             "data/mscoco/mean_std.pth",
                                             train=False,
                                             single_person=False,
                                             img_res=self.hparams["model"]["inp_res_ms"],
                                             mask_res=self.hparams["model"]["out_res"],
                                             kpmap_res=self.hparams["model"]["out_res"],
                                             kpmap_select="all_ex",
                                             keypoint_res=self.hparams["model"]["out_res"],
                                             keypoint_extender=keypoint_extender,
                                             keypoint_label_outsider=True,
                                             keypoint_filter=True)

        self.parser = AE2DParser(PAIR, DISTANCE_MAT, "experiments/mean_length.npy", "experiments/std_length.npy", "experiments/norm_std_length.npy", detection_thres=self.hparams["eval"]["detection_thres"], group_thres=self.hparams["eval"]["group_thres"], max_num_people=self.hparams["model"]["max_num_people"])

        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

    def epoch(self, epoch):
        self.hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, self.hparams['learning_rate'], self.hparams['schedule'], self.hparams['lr_gamma'])

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
            det_map = output_var[-1].data[:, :self.num_parts].cpu().numpy()
            field_map = output_var[-1].data[:, self.num_parts:].cpu().numpy()
            return det_map, field_map

        heatmaploss = self.criterion[0]
        ae2dloss = self.criterion[1]

        imgs = batch["img"]
        det_map_gt = batch["keypoint_map"]
        det_mask_gt = batch["mask"]
        transform_mats = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        keypoint = batch["keypoint"]
        img_ori_size = batch["img_ori_size"]
        volatile = not train

        ae2dloss.prepare(keypoint, _async=False)

        det_map_gt_var = Variable(det_map_gt.cuda(), volatile=volatile)
        det_map_mask_gt_var = Variable(det_mask_gt.cuda(), volatile=volatile)

        output_var = self.model(Variable(imgs[0], volatile=volatile))
        heatmap_var = map(lambda out_stack: out_stack[:, :self.num_parts], output_var)
        emmap_var = map(lambda out_stack: out_stack[:, self.num_parts:], output_var)

        kplosses = []
        emlosses_push = []
        emlosses_pull = []
        for istack in range(self.hparams["model"]["nstack"]):
            kplosses.append(heatmaploss(heatmap_var[istack], det_map_gt_var, det_map_mask_gt_var))
            eml_push, eml_pull = ae2dloss(emmap_var[istack])
            emlosses_push.append(eml_push)
            emlosses_pull.append(eml_pull)

        kploss = sum(kplosses) / len(kplosses)
        emloss_push = sum(emlosses_push) / len(emlosses_push)
        emloss_pull = sum(emlosses_pull) / len(emlosses_pull)

        # TODO: Temporaryly disable to check mseloss integrity
        loss = kploss * self.hparams["model"]["loss_det_cof"] + emloss_push * self.hparams["model"]["loss_em_push_cof"] + emloss_pull * self.hparams["model"]["loss_em_pull_cof"]

        if (loss.data != loss.data).any():
            import pdb; pdb.set_trace()
            # raise RuntimeError("loss is nan")

        image_ids = list(batch["img_index"])
        ans = None
        # TODO: Temporaryly disable
        if not train and True:
            max_map_i = np.argmax([img.size()[-1] for img in imgs])
            max_map_res = imgs[max_map_i].size()[-1] // FACTOR
            det_map_list = list()
            field_map = None
            for scale_i in range(0, len(imgs)):
                if scale_i > 0:
                    output_var = self.model(Variable(imgs[scale_i], volatile=True))
                    det_map, _ = extract_map(output_var)
                else:
                    det_map, field_map = extract_map(output_var)
                output_var = self.model(Variable(imgs[scale_i][..., torch.arange(start=imgs[scale_i].size(-1)-1, end=-1, step=-1).long()], volatile=True))
                det_map_r, _ = extract_map(output_var)
                det_map += det_map_r[..., ::-1][:, datasets.mscoco.FLIP_INDEX + [17]]
                det_map_list.append(batch_resize(det_map, (max_map_res, max_map_res)))

            det_map = np.mean(det_map_list, axis=0) / 2.
            field_map = batch_resize(field_map, (max_map_res, max_map_res))

            transform_mats_max_map = [tfm[max_map_i] for tfm in transform_mats]
            if "no_parse" in config.__dict__ and config.no_parse:
                ans = None
            else:
                ans = parse_result(self.parser, det_map, field_map, self.num_parts-1, image_ids, transform_mats_max_map, img_flipped, img_ori_size)

            if "save_pred" in config.__dict__ and config.save_pred:
                save_args = [
                    det_map,
                    field_map,
                    self.num_parts-1,
                    image_ids,
                    transform_mats_max_map,
                    img_flipped,
                    img_ori_size
                ]
                import os
                if not os.path.isdir(config.save_pred_path):
                    os.makedirs(config.save_pred_path)
                np.save(os.path.join(config.save_pred_path, "pred_{}_{}.npy".format(detail["epoch"], detail["iter"])), save_args)

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
            "pred": None,
            "img_index": image_ids,
            "annotate": ans
        }
    
        return result

def parse_result(parser, det_map, field_map, preserve_num_parts, image_ids, transform_mats, img_flipped, img_ori_size, use_joint_det_scores=False, output_ans=True):
    preds, joint_scores = parser.parse(det_map, field_map)
    preds_new = []
    scores = []
    for isamp, pred_samp in enumerate(preds):
        pred_samp = pred_samp[:, :preserve_num_parts]
        inverse_mat = np.linalg.pinv(transform_mats[isamp])[:2]
        pred_samp[:, :, :2] = kpt_affine(pred_samp[:, :, :2] * FACTOR, inverse_mat)
        if img_flipped[isamp]:
            pred_samp = fliplr_pts(pred_samp, datasets.mscoco.FLIP_INDEX, width=img_ori_size[isamp, 0])

        preds_new.append(pred_samp)
        if use_joint_det_scores:
            joint_scores_samp = joint_scores[isamp][:, :preserve_num_parts]
            # joint_scores_samp[pred_samp[:, :, 2] > 0] += 0.2
            scores.append(joint_scores_samp.mean(axis=1))
        else:
            scores.append(pred_samp[:, :, 2].mean(axis=1).astype(np.float32))

    if output_ans:
        ans = generate_ans(image_ids, preds_new, scores)
        return ans
    else:
        return preds_new

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

        loss_push = (em_cos[push].sum() / self.push_count) if self.push_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)
        loss_pull = (1-(em_cos[pull].sum() / self.pull_count)) if self.pull_count > 0 else torch.autograd.Variable(torch.zeros(1).cuda(), requires_grad=False)

        return loss_push, loss_pull

def Log(logstr):
    if False:
        print(logstr)

class PairCollection(object):
    class DictList(object):
        def __init__(self):
            self._arrdict = dict()
        def __getitem__(self, key):
            if key not in self._arrdict:
                self._arrdict[key] = list()
            return self._arrdict[key]
        def __delitem__(self, key):
            if isinstance(key, int) or (isinstance(key, tuple) and len(key) == 1):
                if key in self._arrdict:
                    del self._arrdict[key]
            elif isinstance(key, tuple) and len(key) == 2:
                if key[0] in self._arrdict and key[1] < len(self._arrdict[key[0]]):
                    del self._arrdict[key[0]][key[1]]
            else:
                raise ValueError()

    def __init__(self):
        self.collection = []
        self.collection_reverse_dict = PairCollection.DictList()

    def append(self, elem1, elem2):
        assert elem1 != -1 and elem2 != -1
        self.collection.append((elem1, elem2))
        self.collection_reverse_dict[elem1].append((len(self.collection) - 1, 0))
        self.collection_reverse_dict[elem2].append((len(self.collection) - 1, 1))

    def contain(self, elem1, elem2):
        assert elem1 != -1 and elem2 != -1
        try:
            self.index(elem1, elem2)
            return True
        except ValueError:
            return False

    def __contains__(self, elems):
        assert len(elems) == 2
        return self.contain(elems[0], elems[1])

    def index(self, elem1, elem2):
        assert elem1 != -1 and elem2 != -1
        inds = self.collection_reverse_dict[elem1]
        for col_ind, pair_ind in inds:
            if self.collection[col_ind][1-pair_ind] == elem2:
                return col_ind
        raise ValueError("Not exist")

    def modify(self, src_elem, dest_elem):
        assert src_elem != dest_elem
        inds_src = self.collection_reverse_dict[src_elem]
        for col_ind, pair_ind in inds_src:
            oppo_elem = self.collection[col_ind][1-pair_ind]
            assert oppo_elem != -1
            assert dest_elem != oppo_elem

            if (dest_elem, oppo_elem) not in self:
                if pair_ind == 0:
                    self.collection[col_ind] = (dest_elem, oppo_elem)
                    self.collection_reverse_dict[dest_elem].append((col_ind, 0))
                else:
                    self.collection[col_ind] = (oppo_elem, dest_elem)
                    self.collection_reverse_dict[dest_elem].append((col_ind, 1))
            else:
                rev_ind_rem = -1
                for col_rev_ind, (col_ind_oppo, pair_ind_oppo) in enumerate(self.collection_reverse_dict[oppo_elem]):
                    if col_ind_oppo == col_ind:
                        rev_ind_rem = col_rev_ind
                        break
                assert rev_ind_rem != -1
                del self.collection_reverse_dict[oppo_elem, rev_ind_rem]
                self.collection[col_ind] = (-1, -1)
        del self.collection_reverse_dict[src_elem]

class AE2DParser(object):
    def __init__(self, pair, joint_dis, joint_spdis_mean_file, joint_spdis_std_file, joint_norm_spdis_std_file, detection_thres=0.1, group_thres=0.1, max_num_people=30):
        self.pair = pair
        self.joint_dis = joint_dis
        self.detection_thres = detection_thres
        self.group_thres = group_thres
        self.joint_spdis_mean = np.load(joint_spdis_mean_file)
        self.joint_spdis_std = np.load(joint_spdis_std_file)
        self.joint_norm_spdis_std = np.load(joint_norm_spdis_std_file)
        self.max_num_people = max_num_people
        self.pool = torch.nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        # suppose det is a tensor
        assert isinstance(det, torch.autograd.Variable)
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det.data

    def parse(self, det, field, combine_discrete=True):
        def debug_draw_pose(img, iperson_ijoint_2_idet, topkind_samp_thr, num_joint):
            iperson_ijoint_2_idet = np.array(iperson_ijoint_2_idet, dtype=np.int32)
            pred = np.zeros(iperson_ijoint_2_idet.shape + (3,), dtype=np.int32)
            if iperson_ijoint_2_idet.shape[0] > 0:
                for ijoint in range(num_joint):
                    idet_2_ind = topkind_samp_thr[ijoint]
                    if idet_2_ind.shape[0] == 0:
                        continue
                    sel_valid_joint_person = (iperson_ijoint_2_idet[:, ijoint] >= 0).nonzero()[0]
                    ind = idet_2_ind[iperson_ijoint_2_idet[sel_valid_joint_person, ijoint]]

                    pred[sel_valid_joint_person, ijoint, 0] = ind % width
                    pred[sel_valid_joint_person, ijoint, 1] = ind / width
                    pred[sel_valid_joint_person, ijoint, 2] = 1

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            import cv2
            def draw_pose(img, pose, factor=1):
                if torch.is_tensor(pose):
                    pose = pose.numpy()

                if pose.shape[1] == 17:
                    pose = np.concatenate((pose, np.zeros((pose.shape[0], 1, 3), dtype=np.float32)), axis=1)
                    pose[:, 17, :2] = (pose[:, 5, :2] + pose[:, 6, :2]) / 2
                    pose[:, 17, 2] = ((pose[:, 5, 2] > 0) & (pose[:, 6, 2] > 0)).astype(np.float32)

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
            
            newimg = img.copy()
            draw_pose(newimg, pred, factor=FACTOR)
            fig, ax = plt.subplots(1, 1, figsize=(40, 40), gridspec_kw={"left": 0, "top": 1, "right": 1, "bottom": 0, "hspace":0, "wspace": 0})
            ax.imshow(newimg)
            plt.show()

        def joint_distance(iperson1, iperson2, pred_valid_mask):
            assert not (pred_valid_mask[iperson1] & pred_valid_mask[iperson2]).any()
            assert pred_valid_mask[iperson1].any()
            assert pred_valid_mask[iperson2].any()
            ijoints1 = pred_valid_mask[iperson1].nonzero()[0]
            ijoints2 = pred_valid_mask[iperson2].nonzero()[0]
            dis = self.joint_dis[ijoints1][:, ijoints2]
            dismin = dis.min()
            dismin_joint1_index, dismin_joint2_index = (dis == dismin).nonzero()
            dismin_joint = list(zip(ijoints1[dismin_joint1_index], ijoints2[dismin_joint2_index]))
            assert dismin > 0
            return dismin, dismin_joint

        def get_joint_connectivity(ijoint1, ijoint2, pred_valid_mask_person, joint_dis):
            assert pred_valid_mask_person[ijoint1] and pred_valid_mask_person[ijoint2]
            joint_dis_clone = joint_dis.copy()
            
            joint_dis_clone[ijoint1, ijoint2] = 100
            joint_dis_clone[ijoint2, ijoint1] = 100

            def spread_connectivity(ijoint):
                spread_mat_joint = np.zeros(joint_dis_clone.shape[0], dtype=bool)
                next_joints = [ijoint]
                while len(next_joints) > 0:
                    spread_mat_joint[next_joints] = 1
                    next_joints = np.nonzero(((joint_dis_clone[next_joints] == 1).astype(np.int32).sum(axis=0) > 0) & pred_valid_mask_person & (~spread_mat_joint))[0]

                return spread_mat_joint

            spread_mat_joint1 = spread_connectivity(ijoint1)
            spread_mat_joint2 = spread_connectivity(ijoint2)

            return (spread_mat_joint1 & spread_mat_joint2).any(), spread_mat_joint1, spread_mat_joint2, joint_dis_clone

        def get_person(iperson_ijoint_2_idet, topkind_samp_thr, iperson, ijoint_sel):
            kp = np.zeros((len(ijoint_sel), 3), dtype=np.float32)
            for ijoint in ijoint_sel:
                idet = iperson_ijoint_2_idet[iperson][ijoint]
                if idet >= 0:
                    ind = topkind_samp_thr[ijoint][idet]
                    kp[ijoint, 0] = ind % width
                    kp[ijoint, 1] = ind / width
                    kp[ijoint, 2] = 1
            return kp

        def get_person_scale(person_kp, joint_spdis_mean, joint_spdis_std):
            scale = 0.
            scale_denom = 0.
            scale_count = 0
            ijoints1, ijoints2 = np.nonzero(joint_spdis_std > 0)
            # sortind = np.argsort(joint_spdis_std[ijoints1, ijoints2])
            # ijoints1 = ijoints1[sortind]
            # ijoints2 = ijoints2[sortind]
            for ijoint1, ijoint2 in zip(ijoints1, ijoints2):
                if person_kp[ijoint1, 2] > 0 and person_kp[ijoint2, 2] > 0:
                    scale += np.linalg.norm(person_kp[ijoint1, :2] - person_kp[ijoint2, :2]) / joint_spdis_mean[ijoint1, ijoint2] / (joint_spdis_std[ijoint1, ijoint2] / joint_spdis_mean[ijoint1, ijoint2])
                    scale_denom += 1 / (joint_spdis_std[ijoint1, ijoint2] / joint_spdis_mean[ijoint1, ijoint2])
                    scale_count += 1
            if scale_denom > 0:
                return (scale / scale_denom), scale_count
            else:
                return None, None

        def get_person_outside_ipair(person_kp, pair, joint_spdis_mean, joint_spdis_std, joint_norm_spdis_std):
            outsider = []
            scale = 0.
            scale_norm = 0.
            scale_counter = 0
            for ipair, (ijoint1, ijoint2) in enumerate(pair):
                if ijoint1 > ijoint2:
                    tmp = ijoint1
                    ijoint1 = ijoint2
                    ijoint2 = tmp
                if person_kp[ijoint1, 2] > 0 and person_kp[ijoint2, 2] > 0:
                    scale += np.linalg.norm(person_kp[ijoint1, :2] - person_kp[ijoint2, :2]) / joint_spdis_mean[ijoint1, ijoint2] / (joint_spdis_std[ijoint1, ijoint2] / joint_spdis_mean[ijoint1, ijoint2])
                    scale_norm += 1. / (joint_spdis_std[ijoint1, ijoint2] / joint_spdis_mean[ijoint1, ijoint2])
                    scale_counter += 1

            if scale_counter > 5 and scale > 0 and scale_norm > 0:
                scale = scale / scale_norm

                for ipair, (ijoint1, ijoint2) in enumerate(pair):
                    if ijoint1 > ijoint2:
                        tmp = ijoint1
                        ijoint1 = ijoint2
                        ijoint2 = tmp
                    if person_kp[ijoint1, 2] > 0 and person_kp[ijoint2, 2] > 0:
                        size_norm = np.linalg.norm(person_kp[ijoint1, :2] - person_kp[ijoint2, :2]) / scale
                        if size_norm < joint_spdis_mean[ijoint1, ijoint2] - 5*joint_norm_spdis_std[ijoint1, ijoint2] or size_norm > joint_spdis_mean[ijoint1, ijoint2] + 5*joint_norm_spdis_std[ijoint1, ijoint2]:
                            outsider.append(ipair)
            return outsider

        assert isinstance(det, np.ndarray) and isinstance(field, np.ndarray)
        num_samp = det.shape[0]
        num_joint = det.shape[1]
        height = det.shape[2]
        width = det.shape[3]
        num_channel = len(self.pair) * 2
        det_nms = self.nms(torch.autograd.Variable(torch.from_numpy(det), volatile=True))

        joint_dis_max = self.joint_dis.max()
        assert joint_dis_max > 3
        # thres = [0] * joint_dis_max
        # thres[1] = self.fallback_group_thres1
        # thres[2] = self.fallback_group_thres2
        # for distance in range(3, len(thres)+1):
        #     thres[distance-1] = self.fallback_group_thres3

        topkval, topkind = det_nms.view(det_nms.size()[:2] + (-1,)).topk(self.max_num_people, dim=-1)
        det_nms = det_nms.numpy()
        topkval = topkval.numpy()
        topkind = topkind.numpy().astype(np.int32)
        # topkloc = np.stack([topkind % width, topkind / width], axis=-1)

        mask_topk_thr = (topkval > self.detection_thres)
        field = field.reshape(field.shape[:2] + (-1,))

        preds = []
        scores = []
        for isamp in range(num_samp):
            topkind_samp_thr = list()
            ijoint_idet_2_iperson = list()
            iperson_ijoint_2_idet = list()
            person_counter = 0
            for ijoint in range(num_joint):
                sel_topk_joint_thr = mask_topk_thr[isamp, ijoint].nonzero()[0]
                topkind_samp_thr.append(topkind[isamp, ijoint][sel_topk_joint_thr])
                ijoint_idet_2_iperson.append([-1] * topkind_samp_thr[ijoint].shape[0])

            sim_neg_all = list()
            for ipair, pr in enumerate(self.pair):
                ijoint1 = pr[0]
                ijoint2 = pr[1]

                topkind_joint1 = topkind_samp_thr[ijoint1]
                topkind_joint2 = topkind_samp_thr[ijoint2]

                if topkind_joint1.shape[0] == 0 or topkind_joint2.shape[0] == 0:
                    sim_neg_all.append(np.zeros((0, 0), dtype=np.float32))
                    continue

                x1 = field[isamp, ipair*2, topkind_joint1][:, np.newaxis]
                y1 = field[isamp, ipair*2+1, topkind_joint1][:, np.newaxis]
                mag1 = np.sqrt(x1 ** 2 + y1 ** 2)

                x2 = field[isamp, ipair*2, topkind_joint2][np.newaxis, :]
                y2 = field[isamp, ipair*2+1, topkind_joint2][np.newaxis, :]
                mag2 = np.sqrt(x2 ** 2 + y2 ** 2)

                mag_mul = mag1 * mag2

                det_mag1 = topkval[isamp, ijoint1][mask_topk_thr[isamp, ijoint1].nonzero()[0]]
                det_mag2 = topkval[isamp, ijoint2][mask_topk_thr[isamp, ijoint2].nonzero()[0]]

                det_mag_mul = det_mag1[:, np.newaxis] * det_mag2[np.newaxis, :]

                # TODO: situation: magnitude too small
                sim_neg = (-(x1 * x2 + y1 * y2) / mag_mul + 1) / 2

                sim_neg_all.append(sim_neg)


            blocked_pair = PairCollection()
            blocked_person_pair = PairCollection()
            # Loop for connecting and disconnecting
            CONNECT_TIMES = 2
            for iconndis in range(CONNECT_TIMES):
                for ipair, pr in enumerate(self.pair):
                    ijoint1 = pr[0]
                    ijoint2 = pr[1]

                    pairs_det = []
                    sim_neg_clone = sim_neg_all[ipair].copy()
                    if sim_neg_clone.shape[0] == 0:
                        continue

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
                        if ((ijoint1, idet1), (ijoint2, idet2)) in blocked_pair:
                            Log("Blocked (%d, %d, %d, %d)" % (ijoint1, idet1, ijoint2, idet2))
                            continue

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
                            iperson_ijoint_2_idet[person_counter][ijoint1] = idet1
                            iperson_ijoint_2_idet[person_counter][ijoint2] = idet2
                            person_counter += 1
                        elif iperson1 != iperson2:
                            if (iperson1, iperson2) in blocked_person_pair:
                                Log("Blocked person (%d, %d)" % (iperson1, iperson2))
                                continue
                            # combine
                            person1 = np.array(iperson_ijoint_2_idet[iperson1], dtype=np.int32)
                            person2 = np.array(iperson_ijoint_2_idet[iperson2], dtype=np.int32)
                            mask_person2 = (person2 >= 0)
                            if ((person1 >= 0) & mask_person2).any():
                                # TODO: conflict
                                pass
                            else:
                                Log("Joining %d and %d" % (iperson1, iperson2))
                                person1[mask_person2] = person2[mask_person2]
                                iperson_ijoint_2_idet[iperson1] = person1.tolist()
                                iperson_ijoint_2_idet[iperson2] = [-1] * num_joint
                                for _ijoint_person2, _idet_person2 in enumerate(person2):
                                    if _idet_person2 >= 0:
                                        ijoint_idet_2_iperson[_ijoint_person2][_idet_person2] = iperson1

                                blocked_person_pair.modify(iperson2, iperson1)
                    
                breaked_any = False
                # Loop for break too long person parts
                while True:
                    if iconndis == CONNECT_TIMES - 1:
                        break
                    # 'breaked' is used for break to this loop and check for if continue this loop
                    breaked = False
                    for iperson in range(len(iperson_ijoint_2_idet)):
                        person_kp = get_person(iperson_ijoint_2_idet, topkind_samp_thr, iperson, list(range(num_joint)))
                        outsider = get_person_outside_ipair(person_kp, self.pair, self.joint_spdis_mean, self.joint_spdis_std, self.joint_norm_spdis_std)
                        joint_dis = self.joint_dis
                        for ipair in outsider:
                            ijoint_l, ijoint_r = self.pair[ipair]
                            idet_l, idet_r = iperson_ijoint_2_idet[iperson][ijoint_l], iperson_ijoint_2_idet[iperson][ijoint_r]
                            if ((ijoint_l, idet_l), (ijoint_r, idet_r)) not in blocked_pair:
                                blocked_pair.append((ijoint_l, idet_l), (ijoint_r, idet_r))
                                Log("Blocking (%d, %d, %d, %d)" % (ijoint_l, idet_l, ijoint_r, idet_r))
                            conn, mask_joint1, mask_joint2, joint_dis = get_joint_connectivity(ijoint_l, ijoint_r, (np.array(iperson_ijoint_2_idet[iperson]) >= 0), joint_dis)
                            if not conn:
                                iperson_ijoint_2_idet.append([-1] * num_joint)
                                for ijoint in np.nonzero(mask_joint2)[0]:
                                    idet = iperson_ijoint_2_idet[iperson][ijoint]
                                    iperson_ijoint_2_idet[person_counter][ijoint] = idet
                                    iperson_ijoint_2_idet[iperson][ijoint] = -1
                                    ijoint_idet_2_iperson[ijoint][idet] = person_counter
                                Log("Breaking %d to %d and %d" % (iperson, iperson, person_counter))
                                blocked_person_pair.append(iperson, person_counter)
                                
                                Log("Blocking person (%d, %d)" % (iperson, person_counter))
                                person_counter += 1
                                breaked = True
                                breaked_any = True
                                break
                        if breaked:
                            break

                    if not breaked:
                        break
                if not breaked_any:
                    break

            # allocate for remaining detected point a new person
            for ijoint in range(num_joint):
                for idet, iperson in enumerate(ijoint_idet_2_iperson[ijoint]):
                    if iperson == -1:
                        ijoint_idet_2_iperson[ijoint][idet] = person_counter
                        iperson_ijoint_2_idet.append([-1] * num_joint)
                        iperson_ijoint_2_idet[person_counter][ijoint] = idet
                        person_counter += 1

            # after the iperson_ijoint_2_idet is compressed, the iperson in ijoint_idet_2_iperson will be not valid
            del ijoint_idet_2_iperson

            # compress
            iperson_ijoint_2_idet = np.array(iperson_ijoint_2_idet, dtype=np.int32).reshape((-1, num_joint))

            # 1. Get num of joints of each person
            # 2. Sort person by descending num of valid joints
            # 3. Select person with most joints (most <= num_joint - 3)
            # 4. Start from dis = 2
            # 5. Find person that distance == dis, within space distance thres
            #    (if distance = 1 and field force sim is within group_thres, volating assert)
            # 6. Find most closed part, if no, next, else, merge them and goto 1
            # 7. check `dis` integrity, if not satified, select next person and goto 4, else, goto 5

            if combine_discrete:
                while True:
                    pred_valid_mask = (iperson_ijoint_2_idet >= 0)
                    num_joint_pperson = pred_valid_mask.astype(np.int32).sum(axis=1)
                    sel_person = (num_joint_pperson > 0).nonzero()[0]

                    pred_valid_mask = pred_valid_mask[sel_person]
                    iperson_ijoint_2_idet = iperson_ijoint_2_idet[sel_person]
                    num_joint_pperson = num_joint_pperson[sel_person]

                    if iperson_ijoint_2_idet.shape[0] == 0:
                        break

                    def find_joinable_person():
                        for iperson1 in np.argsort(num_joint_pperson):
                            sel_person_compl = ((pred_valid_mask[[iperson1]] & pred_valid_mask).astype(np.int32).sum(axis=1) == 0).nonzero()[0]
                            dis_2_person = [[] for _ in range(joint_dis_max)]
                            for iperson2 in sel_person_compl:
                                dis_min, pairs_ijoint_dis_min = joint_distance(iperson1, iperson2, pred_valid_mask)
                                dis_2_person[dis_min-1].append((iperson2, pairs_ijoint_dis_min))

                            person1 = get_person(iperson_ijoint_2_idet, topkind_samp_thr, iperson1, list(range(num_joint)))
                            # count start from 1, and persons with distance 1 should already be merged
                            for distance in range(2, joint_dis_max+1):
                                min_space_dis = width
                                iperson_min_space_dis = -1
                                for iperson2, pairs_ijoint in dis_2_person[distance-1]:
                                    person2 = get_person(iperson_ijoint_2_idet, topkind_samp_thr, iperson2, list(range(num_joint)))
                                    scale1, num_pair_p1 = get_person_scale(person1, self.joint_spdis_mean, self.joint_spdis_std)
                                    scale2, num_pair_p2 = get_person_scale(person2, self.joint_spdis_mean, self.joint_spdis_std)
                                    if scale1 and scale2:
                                        num_pair_total = (self.joint_spdis_std.size - self.joint_spdis_std.shape[0]) / 2
                                        if (4 * (num_pair_p1 * num_pair_p2) / float(num_pair_total)**2) * max(scale1, scale2) / min(scale1, scale2) > 1.5:
                                            continue
                                        scale_mean = (scale1 * num_pair_p1 + scale2 * num_pair_p2) / (num_pair_p1 + num_pair_p2)
                                    elif scale1:
                                        scale_mean = scale1
                                    elif scale2:
                                        scale_mean = scale2
                                    else:
                                        scale_mean = None

                                    if scale_mean and scale_mean > 0:
                                        for pr_ijoint in pairs_ijoint:
                                            point1 = person1[pr_ijoint[0]][:2]
                                            point2 = person2[pr_ijoint[1]][:2]
                                            space_dis = np.linalg.norm(point1 - point2)
                                            space_dis_mean = self.joint_spdis_mean[min(pr_ijoint[0], pr_ijoint[1]), max(pr_ijoint[0], pr_ijoint[1])]
                                            space_norm_dis_std = self.joint_norm_spdis_std[min(pr_ijoint[0], pr_ijoint[1]), max(pr_ijoint[0], pr_ijoint[1])]

                                            if (space_dis / scale_mean) >= (space_dis_mean - space_norm_dis_std * 2) and (space_dis / scale_mean) <= (space_dis_mean + space_norm_dis_std * 2):
                                                # if space_dis < min_space_dis:
                                                #     min_space_dis = space_dis
                                                #     iperson_min_space_dis = iperson2
                                                return (iperson1, iperson2)

                                # if iperson_min_space_dis != -1:
                                #     return (iperson1, iperson_min_space_dis)

                        return None

                    ipersons = find_joinable_person()
                    if ipersons is not None:
                        iperson1, iperson2 = ipersons
                        person1 = np.array(iperson_ijoint_2_idet[iperson1], dtype=np.int32)
                        person2 = np.array(iperson_ijoint_2_idet[iperson2], dtype=np.int32)
                        mask_person2 = (person2 >= 0)
                        assert not ((person1 >= 0) & mask_person2).any()
                        person1[mask_person2] = person2[mask_person2]
                        iperson_ijoint_2_idet[iperson1] = person1.tolist()
                        iperson_ijoint_2_idet[iperson2] = [-1] * num_joint
                    else:
                        break

            # compress
            sel_person = ((iperson_ijoint_2_idet >= 0).astype(np.int32).sum(axis=1) > 0).nonzero()[0]
            num_person = sel_person.shape[0]
            iperson_ijoint_2_idet = iperson_ijoint_2_idet[sel_person]

            # convert iperson_ijoint_2_idet to prediction
            pred = np.zeros(iperson_ijoint_2_idet.shape + (3,), dtype=np.int32)
            score = np.zeros(iperson_ijoint_2_idet.shape, dtype=np.float32)
            if iperson_ijoint_2_idet.shape[0] > 0:
                for ijoint in range(num_joint):
                    idet_2_ind = topkind_samp_thr[ijoint]
                    if idet_2_ind.shape[0] == 0:
                        continue
                    sel_valid_joint_person = (iperson_ijoint_2_idet[:, ijoint] >= 0).nonzero()[0]
                    ind = idet_2_ind[iperson_ijoint_2_idet[sel_valid_joint_person, ijoint]]

                    pred_x = ind % width
                    pred_y = ind / width
                    pred[sel_valid_joint_person, ijoint, 0] = pred_x
                    pred[sel_valid_joint_person, ijoint, 1] = pred_y
                    pred[sel_valid_joint_person, ijoint, 2] = 1
                    score[sel_valid_joint_person, ijoint] = det[isamp, ijoint][pred_y, pred_x]

            assert (score >= 0).all()
            scores.append(score)

            pred = pred.astype(np.float32)
            for iperson in range(pred.shape[0]):
                for ijoint in range(pred.shape[1]):
                    if pred[iperson, ijoint, 2] > 0:
                        tmp = det[isamp, ijoint]
                        x = int(pred[iperson, ijoint, 0])
                        y = int(pred[iperson, ijoint, 1])
                        x_bias = 0.0
                        y_bias = 0.0
                        y_diff = tmp[min(y+1, tmp.shape[0]-1), x] - tmp[max(y-1, 0), x]
                        if y_diff > 0:
                            y_bias += 0.25
                        elif y_diff < 0:
                            y_bias -= 0.25

                        x_diff = tmp[y, min(x+1, tmp.shape[1]-1)] - tmp[y, max(x-1, 0)]
                        if x_diff > 0:
                            x_bias += 0.25
                        elif x_diff < 0:
                            x_bias -= 0.25
                        pred[iperson, ijoint, 0] = float(x) + x_bias + 0.5
                        pred[iperson, ijoint, 1] = float(y) + y_bias + 0.5

            preds.append(pred)

        return preds, scores

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
                        tmp["keypoints"] += [float(p[0]), float(p[1]), 0]
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
    def __init__(self, img_index=None):
        hparams = dict(
            learning_rate=2e-4,
            weight_decay=0,
            model=dict(
                nstack=4,
                hg_dim=256,
                increase=128,
                inp_res=512,
                inp_res_ms=[512],#[512, 256, 1024]
                out_res=128,
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
        if img_index is None:
            self.dataloader = torch.utils.data.DataLoader(self.exp.val_dataset,
                                                        batch_size=32,
                                                        shuffle=False,
                                                        num_workers=0,
                                                        collate_fn=self.exp.test_collate_fn,
                                                        pin_memory=True)
            self.data = next(iter(self.dataloader))
        else:
            img_sel_index = self.exp.val_dataset.get_index_by_imgidx(img_index)
            if img_sel_index == -1:
                raise RuntimeError()
            self.data = self.exp.test_collate_fn([self.exp.val_dataset[img_sel_index]])
        # self.pair = [[0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [5,6],
        #              [5,7], [7,9], [6,8], [8,10], [5,11], [6,12], [11,12],
        #              [11,13], [13,15], [12,14], [14,16]]
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

        det_map = det_map.numpy()

        preds, joint_scores = self.exp.parser.parse(det_map, field_map)
        preds_new = []
        scores = []
        for isamp, pred_samp in enumerate(preds):
            if transform_back:
                pred_samp = pred_samp[:, :17]
                inverse_mat = np.linalg.pinv(transform_mats[isamp][0])[:2]
                pred_samp[:, :, :2] = kpt_affine(pred_samp[:, :, :2] * FACTOR, inverse_mat)
                if self.data["img_flipped"][isamp]:
                    pred_samp = fliplr_pts(pred_samp, datasets.mscoco.FLIP_INDEX, width=self.data["img_ori_size"][isamp, 0])
            else:
                pred_samp[:, :, :2] = pred_samp[:, :, :2] * FACTOR
            preds_new.append(pred_samp)
            scores.append(pred_samp[:, :, 2].mean(axis=1).astype(np.float32))
        # print("Keypoint Original:")
        # print(self.data["keypoint_ori"])
        # print("\nKeypoint Predicted:")
        # print(preds)
        # print("\nScore:")
        # print(scores)
        if transform_back:
            keypoint = [
                kp.numpy() if kp is not None else np.zeros((0, 17, 3), dtype=np.float32) for kp in self.data["keypoint_ori"]
            ]
        else:
            keypoint = [
                np.concatenate((kp.numpy()[:, :, :2] * FACTOR, kp.numpy()[:, :, [2]]), axis=2) if kp is not None else np.zeros((0, 17, 3), dtype=np.float32) for kp in keypoint
            ]
        return {
            "pred": preds_new,
            "score": scores,
            "image_id": image_ids,
            "det_map": det_map,
            "field_map": field_map,
            "keypoint": keypoint,
            "img": self.data["img"][0].numpy()
        }

    def plot(self, transformed_space=True):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import cv2
        def draw_pose(img, pose, factor=1):
            if torch.is_tensor(pose):
                pose = pose.numpy()

            if pose.shape[1] == 17:
                pose = np.concatenate((pose, np.zeros((pose.shape[0], 1, 3), dtype=np.float32)), axis=1)
                pose[:, 17, :2] = (pose[:, 5, :2] + pose[:, 6, :2]) / 2
                pose[:, 17, 2] = ((pose[:, 5, 2] > 0) & (pose[:, 6, 2] > 0)).astype(np.float32)

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

        if transformed_space:
            result = self.generate_output(transform_back=False)
            field_map = result["field_map"]
            img = self.exp.val_dataset.restore_image(result["img"])
        else:
            result = self.generate_output(transform_back=True)
            img = [self.exp.val_dataset._load_image(img_idx, bgr=False) for img_idx in result["image_id"]]

        preds = result["pred"]
        scores = result["score"]
        image_ids = result["image_id"]
        gts = result["keypoint"]

        # gt, pred
        # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20), gridspec_kw={"wspace": 0, "hspace": 0, "left": 0, "top": 1, "right": 1, "bottom": 0})
        num_pair = len(PAIR)
        num_col = 6
        num_row = (num_pair + num_col - 1) // num_col
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            if transformed_space:
                fig = plt.figure(figsize=(num_col * 7, num_col * 7 // 2 + num_row * 7))
                gs = gridspec.GridSpec(2, 1, wspace=0, hspace=0, left=0, top=1, right=1, bottom=0, height_ratios=[num_col / 2., num_row])
                gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0, hspace=0, subplot_spec=gs[0])
            else:
                fig = plt.figure(figsize=(num_col * 7, num_col * 7 // 2))
                gs0 = gridspec.GridSpec(1, 2, wspace=0, hspace=0, left=0, top=1, right=1, bottom=0)

            draw_img_gt = img[i].copy()
            draw_img_pred = draw_img_gt.copy()
            draw_pose(draw_img_gt, gt, 1)
            draw_pose(draw_img_pred, pred, 1)
            ax = fig.add_subplot(gs0[0])
            ax.imshow(draw_img_gt)
            ax.axis("off")
            ax = fig.add_subplot(gs0[1])
            ax.imshow(draw_img_pred)
            ax.axis("off")

            if transformed_space:
                gs1 = gridspec.GridSpecFromSubplotSpec(num_row, num_col, wspace=0.05, hspace=0, subplot_spec=gs[1])
                fm = batch_resize(field_map[i], draw_img_gt.shape[-3:-1])
                for ipair in range(len(PAIR)):
                    img_field = (generate_field_img(fm[ipair*2], fm[ipair*2+1]) * 255).round().clip(0, 255).astype(np.uint8)
                    draw_img_field = cv2.addWeighted(img[i], 1, img_field, 0.8, 0)
                    for joint_person in gt:
                        for ipairjoint in range(2):
                            joint = joint_person[PAIR[ipair][ipairjoint]]
                            if joint[2] > 0:
                                pt = tuple(joint[:2].astype(np.int32).tolist())
                                force = (int(fm[ipair*2, pt[1], pt[0]]*20), int(fm[ipair*2+1, pt[1], pt[0]]*20))
                                cv2.arrowedLine(draw_img_field, pt, (pt[0]+force[0], pt[1]+force[1]), (255, 255, 255), thickness=2)

                    ax = fig.add_subplot(gs1[ipair // num_col, ipair % num_col])
                    ax.imshow(draw_img_field)
                    ax.axis("off")
            plt.show()

    def evaluate(self, mode=None):
        if mode is None:
            mode = "pred"
        assert mode in ["pred", "pred+2", "gt", "gtalign4", "gtalign4+2", "preddetrate"]
        result = self.generate_output()
        preds = result["pred"]
        gt = result["keypoint"]
        scores = result["score"]
        image_ids = result["image_id"]

        if mode == "pred":
            ans = generate_ans(image_ids, preds, scores)
        elif mode == "pred+2":
            preds = [np.concatenate((kp[:, :, :2] + 2, kp[:, :, [2]]), axis=2).astype(np.float32) for kp in preds]
            ans = generate_ans(image_ids, preds, scores)
        elif mode == "preddetrate":
            pred_count = sum([(kp[:, :, 2] > 0).astype(np.int32).sum() for kp in preds])
            gt_count = sum([(kp[:, :, 2] > 0).astype(np.int32).sum() for kp in gt])
            print("pr: %d, gt: %d, rate: %.4f" % (pred_count, gt_count, (float(pred_count) / gt_count)))
            return
        elif mode == "gt":
            scores = [np.ones(kp.shape[0]) for kp in gt]
            for kp in gt:
                kp[:, :, 2] = (kp[:, :, 2] > 0).astype(np.float32)
            ans = generate_ans(image_ids, gt, scores)
        elif mode == "gtalign4":
            scores = [np.ones(kp.shape[0]) for kp in gt]
            gt = [np.concatenate((((kp[:, :, :2] // 4) * 4), (kp[:, :, [2]] > 0).astype(np.float32)), axis=2).astype(np.float32) for kp in gt]
            ans = generate_ans(image_ids, gt, scores)
        elif mode == "gtalign4+2":
            scores = [np.ones(kp.shape[0]) for kp in gt]
            gt = [np.concatenate((((kp[:, :, :2] // 4) * 4) + 2, (kp[:, :, [2]] > 0).astype(np.float32)), axis=2).astype(np.float32) for kp in gt]
            ans = generate_ans(image_ids, gt, scores)
        self.exp.evaluate(image_ids, ans)

def generate_distribution(pair):
    def keypoint_extender(keypoint):
        keypoint = np.concatenate((keypoint, np.zeros((keypoint.shape[0], 1, 3), dtype=np.float32)), axis=1)
        keypoint[:, -1, :2] = (keypoint[:, 5, :2] + keypoint[:, 6, :2]) / 2
        keypoint[:, -1, 2] = ((keypoint[:, 5, 2] > 0) & (keypoint[:, 6, 2] > 0)).astype(np.float32)
        return keypoint
    val_dataset = datasets.COCOPose("data/mscoco/images",
                                    self.coco,
                                    "data/mscoco/split.pth",
                                    "data/mscoco/mean_std.pth",
                                    train=False,
                                    single_person=False,
                                    img_res=[512],
                                    mask_res=0,
                                    kpmap_res=0,
                                    kpmap_select="all_ex",
                                    keypoint_res=512,
                                    keypoint_extender=keypoint_extender,
                                    keypoint_label_outsider=True,
                                    keypoint_filter=True)
    dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=datasets.COCOPose.collate_function,
                                             pin_memory=True)
    # TODO: per connection include_sigma and exclude_sigma

if __name__ == "__main__":
    if sys.argv[1] == "eval":
        TestOutput().evaluate(sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "plot":
        TestOutput().plot(transformed_space=False)
    elif sys.argv[1] == "debug":
        TestOutput(img_index=524420).plot(transformed_space=True)
    elif sys.argv[1] == "generate_distribution":
        generate_distribution()
