from __future__ import print_function, absolute_import
import torch
import pose.models as models
import pose.datasets as datasets
from pose.utils.evaluation import PR_multi
import pose.utils.config as config
from pose.utils.misc import adjust_learning_rate
from pose.utils.transforms import fliplr_chwimg, fliplr_map
from pose.models import HeatmapLoss, AELoss
from pose.utils.group import HeatmapParser

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import numpy as np
import torchvision.utils as vutils

INP_RES = 512
FACTOR = 4
OUT_RES = INP_RES // FACTOR
INP_EXTRA_RES = [INP_RES//2, INP_RES*2]
INP_EVAL_RES = [INP_RES] + INP_EXTRA_RES

"""
TODO:

DEBUG:

1. evaluation
"""


class Experiment(object):
    def __init__(self, hparams):
        self.num_parts = datasets.mscoco.NUM_PARTS
        self.hparams = hparams
        self.model = torch.nn.DataParallel(
            models.PoseHGNet(
                inp_dim=3,
                out_dim=self.num_parts*4, # original ae
                nstack=hparams["model"]["nstack"],
                hg_dim=hparams["model"]["hg_dim"],
                increase=hparams["model"]["increase"],
                bn=hparams["model"]["bn"]).cuda())
    
        self.criterion = torch.nn.ModuleList([
            HeatmapLoss(),
            AELoss()]).cuda()
        
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
                                               keypoint_res=OUT_RES)

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
                                             keypoint_res=OUT_RES)

        self.parser = HeatmapParser(detection_val=0.1, max_num_people=self.hparams["model"]["max_num_people"])

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

    def process(self, batch, train, detail=None):
        def extract_map(output_var):
            det_map = output_var[-1].data[:, :self.num_parts].cpu().numpy().astype(np.float32)
            tag_map = output_var[-1].data[:, self.num_parts:(self.num_parts*2)].cpu().numpy().astype(np.float32)
            return det_map, tag_map

        mse_criterion = self.criterion[0]
        ae_criterion = self.criterion[1]

        imgs = batch["img"]
        kpmap = batch["keypoint_map"]
        keypoint_gt = batch["keypoint"]
        mask = batch["mask"]
        transform_mats = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        volatile = not train
        img_vars = [torch.autograd.Variable(img.cuda(async=True), volatile=volatile) for img in imgs]
        kpmap_var = torch.autograd.Variable(kpmap.cuda(async=True), volatile=volatile)
        mask_var = torch.autograd.Variable(mask.cuda(async=True), volatile=volatile)
        keypoint_gt_ae = torch.zeros(len(keypoint_gt), self.hparams["model"]["max_num_people"], self.num_parts, 2).long()

        for bi, keypoint_gt_i in enumerate(keypoint_gt):
            person_counter = 0
            avail_joint_counter = 0
            if keypoint_gt_i is None:
                continue
            for person_i, person_pts in enumerate(keypoint_gt_i):
                for part_i, pt in enumerate(person_pts):
                    x, y, flag_pt = int(pt[0]), int(pt[1]), int(pt[2])
                    if flag_pt > 0 and x >= 0 and y >= 0 and x < OUT_RES and y < OUT_RES:
                        avail_joint_counter += 1
                        keypoint_gt_ae[bi, person_counter, part_i, 0] = OUT_RES * OUT_RES * part_i + OUT_RES * y + x
                        keypoint_gt_ae[bi, person_counter, part_i, 1] = 1
            if avail_joint_counter > 0:
                person_counter += 1
        keypoint_gt_ae = torch.autograd.Variable(keypoint_gt_ae)

        output_var = self.model(img_vars[0])

        loss_dets = []
        loss_tags = []
        for j in range(0, self.hparams["model"]["nstack"]):
            output_j = output_var[j]
            loss_dets.append(mse_criterion(output_j[:, :self.num_parts], kpmap_var, mask_var))
            loss_tags.append(ae_criterion(output_j[:, self.num_parts:(self.num_parts*2)].contiguous().view(output_j.size()[0], -1, 1), keypoint_gt_ae))

        loss_dets = torch.stack(loss_dets, dim=0)
        loss_tags = torch.stack(loss_tags, dim=0).cuda(output_var[0].get_device())

        loss_det = loss_dets.mean() * self.hparams["model"]["loss_det_cof"]
        loss_tag_push = loss_tags[:, :, 0].mean() * self.hparams["model"]["loss_push_cof"]
        loss_tag_pull = loss_tags[:, :, 1].mean() * self.hparams["model"]["loss_pull_cof"]

        loss = loss_det + loss_tag_push + loss_tag_pull

        phase_str = "train" if train else "valid"
        config.tb_writer.add_scalars(config.exp_name + "/loss_det", {phase_str: loss_det}, detail["step"])
        config.tb_writer.add_scalars(config.exp_name + "/loss_tag_push", {phase_str: loss_tag_push}, detail["step"])
        config.tb_writer.add_scalars(config.exp_name + "/loss_tag_pull", {phase_str: loss_tag_pull}, detail["step"])

        image_ids = None
        ans = None

        if not train:
            max_map_i = np.argmax([img.size()[-1] for img in imgs])
            max_map_res = imgs[max_map_i].size()[-1] // FACTOR
            det_map_list = list()
            tag_map_list = list()
            for scale_i in range(0, len(img_vars)):
                if scale_i > 0:
                    output_var = self.model(img_vars[scale_i])
                det_map, tag_map = extract_map(output_var)
                output_var = self.model(torch.autograd.Variable(img_vars[scale_i].data[..., torch.arange(start=img_vars[scale_i].size(-1)-1, end=-1, step=-1).long().cuda()], volatile=True))
                det_map_r, tag_map_r = extract_map(output_var)
                det_map += det_map_r[..., ::-1][:, datasets.mscoco.FLIP_INDEX]
                if det_map.max() <= 10:
                    det_map_list.append(batch_resize(det_map, (max_map_res, max_map_res)))
                    # if scale is closed to 1.0
                    if abs(INP_EVAL_RES[scale_i] / float(INP_RES) - 1) < 0.5:
                        tag_map_list.append(batch_resize(tag_map, (max_map_res, max_map_res)))
                        tag_map_list.append(batch_resize(tag_map_r[..., ::-1][:, datasets.mscoco.FLIP_INDEX], (max_map_res, max_map_res)))
            if len(det_map_list) > 0 and len(tag_map_list) > 0:
                det_map_mean = np.mean(det_map_list, axis=0) / 2.
                tag_map_cat = np.stack(tag_map_list, axis=-1)

                grouped = self.parser.parse(det_map_mean, tag_map_cat)
                scores = [sample_grouped[:, :, 2].mean(axis=1) if len(sample_grouped) > 0 else 0. for sample_grouped in grouped]

                for batch_i, sample_grouped in enumerate(grouped):
                    inverse_mat = np.linalg.pinv(transform_mats[batch_i][max_map_i])[:2]
                    for person_i in range(len(sample_grouped)):
                        sample_grouped[person_i] = refine(det_map_mean[batch_i],
                                                    tag_map_cat[batch_i],
                                                    sample_grouped[person_i])
                    if len(sample_grouped) > 0:
                        if img_flipped[batch_i]:
                            sample_grouped[:, :, 0] = INP_EVAL_RES[max_map_i] / FACTOR - sample_grouped[:, :, 0]
                            sample_grouped = sample_grouped[:, datasets.mscoco.FLIP_INDEX]
                        sample_grouped[:, :, :2] = kpt_affine(sample_grouped[:, :, :2] * FACTOR, inverse_mat)

                image_ids = list(batch["img_index"])
                ans = generate_ans(image_ids, grouped, scores)
            else:
                print("No avail result")

        # if ("summary" in detail and detail["summary"]):
        #     self.summary_image(img, score_map.cpu(), target, config.exp_name + "/" + ("train" if train else "val"), detail["epoch"] + 1)

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