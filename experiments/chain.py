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

        self.criterion = DataParallelCriterion(
            ChainLoss(self.num_parts,
                      len(PAIR)).cuda(), comp_mean=True)

        self.pfgen = PairwiseFieldGenerator(radius=self.hparams["dataset"]["label_field_radius"], pair=PAIR, pair_indexof=PAIR_INDEXOF, res=OUT_RES)
        
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
                                               custom_generator=self.pfgen)

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
                                             custom_generator=self.pfgen)

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

        imgs = batch["img"]
        det_map = batch["keypoint_map"]
        det_mask = batch["mask"]
        transform_mats = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        field_map = batch["custom"][0]
        field_map_mask = batch["custom"][1]
        volatile = not train

        det_map_var = Variable(det_map, volatile=volatile)
        det_map_mask_var = Variable(det_mask, volatile=volatile)
        field_map_var = Variable(field_map, volatile=volatile)
        field_map_mask_var = Variable(field_map_mask, volatile=volatile)
        num_field = field_map_mask_var.size(1)

        output_var = self.model(Variable(imgs[0], volatile=volatile))

        det_loss, field_loss = self.criterion(output_var, det_map_var, det_map_mask_var, field_map_var, field_map_mask_var)
        loss = det_loss * self.hparams["model"]["loss_det_cof"] + field_loss * self.hparams["model"]["loss_field_cof"]

        if (loss.data != loss.data).any():
            import pdb; pdb.set_trace()
            # raise RuntimeError("loss is nan")

        if not train:
            max_map_i = np.argmax([img.size()[-1] for img in imgs])
            max_map_res = imgs[max_map_i].size()[-1] // FACTOR
            det_map_list = list()
            field_map = None
            for scale_i in range(0, len(imgs)):
                if scale_i > 0:
                    output_var = gather(self.model(Variable(imgs[scale_i], volatile=True)), target_device=-1)
                    det_map, _ = extract_map(output_var)
                else:
                    output_var = gather(output_var, -1)
                    det_map, field_map = extract_map(output_var)
                det_map, _ = extract_map(output_var)
                output_var = gather(self.model(Variable(imgs[scale_i][..., torch.arange(start=imgs[scale_i].size(-1)-1, end=-1, step=-1).long()], volatile=True)), -1)
                det_map_r, _ = extract_map(output_var)
                det_map += det_map_r[..., ::-1][:, datasets.mscoco.FLIP_INDEX]
                if det_map.max() <= 10:
                    det_map_list.append(batch_resize(det_map, (max_map_res, max_map_res)))

            if len(det_map_list) > 0:
                det_map_mean = np.mean(det_map_list, axis=0) / 2.

                grouped = self.parser.parse(det_map_mean, field_map)

                scores = [sample_grouped[:, :, 2].mean(axis=1) if len(sample_grouped) > 0 else 0. for sample_grouped in grouped]

                for batch_i, sample_grouped in enumerate(grouped):
                    inverse_mat = np.linalg.pinv(transform_mats[batch_i][max_map_i])[:2]
                    for person_i in range(len(sample_grouped)):
                        sample_grouped[person_i] = refine(det_map_mean[batch_i],
                                                    field_map_cat[batch_i],
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

        # pred = self.parser.parse(*extract_map(output_var))

        phase_str = "train" if train else "valid"
        config.tb_writer.add_scalars(config.exp_name + "/loss_det", {phase_str: det_loss.data.cpu()[0]}, detail["step"])
        config.tb_writer.add_scalars(config.exp_name + "/loss_field", {phase_str: field_loss.data.cpu()[0]}, detail["step"])

        result = {
            "loss": loss,
            "acc": 0,
            "recall": 0,
            "prec": None,
            "index": batch["index"],
            "pred": None
        }
    
        return result

class PairwiseFieldGenerator(object):
    def __init__(self, radius, pair, pair_indexof, res):
        self.radius = radius
        self.pair = pair
        self.pair_indexof = pair_indexof
        self.res = res
        X, Y = np.meshgrid(np.arange(radius*2+1), np.arange(radius*2+1))
        X -= radius
        Y -= radius
        self.select_X = X.astype(np.int32)
        self.select_Y = Y.astype(np.int32)
        self.select = (np.sqrt(X**2 + Y**2) <= radius)

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def generate(self, img, keypoint, mask):
        num_field = len(self.pair) * 2
        res = self.res
        width = res
        height = res
        radius = self.radius
        out = np.zeros((num_field * 2, res, res), dtype=np.float32)
        out_mask = np.zeros((num_field, res, res), dtype=np.bool)
        dirty_mask = np.zeros((num_field, res, res), dtype=np.bool)
        assert mask.shape[-1] == res

        for iperson, personpt in enumerate(keypoint):
            for ijoint, joint in enumerate(personpt):
                x = int(joint[0])
                y = int(joint[1])
                if joint[2] <= 0 or x < 0 or y < 0 or x >= width or y >= height:
                    continue
                if mask[y, x] != 1:
                    continue
                ul = int(x - radius), int(y - radius)
                br = int(x + radius + 1), int(y + radius + 1)

                c,d = max(0, -ul[0]), min(br[0], width) - ul[0]
                a,b = max(0, -ul[1]), min(br[1], height) - ul[1]
                # cc,dd = max(0, ul[0]), min(br[0], width)
                # aa,bb = max(0, ul[1]), min(br[1], height)

                select_X = self.select_X[a:b, c:d][self.select[a:b, c:d]] + x
                select_Y = self.select_Y[a:b, c:d][self.select[a:b, c:d]] + y

                for ipair, idestend in self.pair_indexof[ijoint]:
                    secondijoint = self.pair[ipair][idestend]
                    secondjoint = personpt[secondijoint]
                    if secondjoint[2] > 0:
                        force = (secondjoint[:2] - joint[:2])
                        force_norm = np.linalg.norm(force)
                        if force_norm < 0.1:
                            continue
                        force /= force_norm
                        ifield = ipair*2+(1-idestend)
                        out[ifield, select_Y, select_X] = force[0]
                        out[num_field+ifield, select_Y, select_X] = force[1]
                        dirty_mask[ifield, select_Y, select_X] = np.logical_or(out_mask[ifield, select_Y, select_X], dirty_mask[ifield, select_Y, select_X])
                        out_mask[ifield, select_Y, select_X] = True

        out_mask = np.logical_and(out_mask, np.logical_not(dirty_mask))
        out_mask = np.logical_and(out_mask, np.expand_dims(mask.astype(np.bool), axis=0))

        return torch.from_numpy(out).float(), torch.from_numpy(out_mask.astype(np.float32)).float()

class ChainLoss(torch.nn.Module):
    def __init__(self, num_parts, num_pairs):
        super(ChainLoss, self).__init__()
        self.num_parts = num_parts
        self.num_pairs = num_pairs
        self.heapmaploss = HeatmapLoss()
        self.fieldmaploss = FieldmapLoss()

    def forward(self, pred, gt_detmap, mask_detmap, gt_fieldmap, mask_fieldmap):
        if not isinstance(pred, list):
            pred = [pred]
        kplosses = []
        fclosses = []
        for pr in pred:
            kplosses.append(self.heapmaploss(pr[:, :self.num_parts], gt_detmap, mask_detmap))
            fclosses.append(self.fieldmaploss(pr[:, self.num_parts:], gt_fieldmap, mask_fieldmap))

        kploss = sum(kplosses) / len(kplosses)
        fcloss = sum(fclosses) / len(fclosses)
        return kploss, fcloss

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

class TestOutput(object):
    def __init__(self):
        hparams = dict(
            learning_rate=5e-4,
            weight_decay=0,
            model=dict(
                nstack=4,
                hg_dim=256,
                increase=128,
                bn=False,
                max_num_people=30,
                loss_det_cof=1,
                loss_field_cof=0.0001),
            dataset=dict(
                label_field_radius=2),
            eval=dict(
                detection_thres=0.1,
                group_thres=0.86)
        )
        self.exp = Experiment(hparams)
        self.dataloader = torch.utils.data.DataLoader(self.exp.val_dataset,
                                                      batch_size=8,
                                                      shuffle=False,
                                                      num_workers=5,
                                                      collate_fn=self.exp.test_collate_fn,
                                                      pin_memory=True)
        self.data = next(iter(self.dataloader))
        print("Data loaded")

    def generate_output(self, transform_back=True):
        image_ids = list(self.data["img_index"])
        det_map = self.data["keypoint_map"]
        det_mask = self.data["mask"]
        field_map = self.data["custom"][0]
        field_map_mask = self.data["custom"][1]
        transform_mats = self.data["img_transform"]
        img_flipped = self.data["img_flipped"]

        preds = self.exp.parser.parse(det_map, field_map)
        preds_new = []
        scores = []
        for batch_i, sample_pred in enumerate(preds):
            sample_pred = sample_pred.numpy()
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
            "keypoint": self.data["keypoint"],
            "img": self.data["img"]
        }

    def plot(self):
        def draw_pose(img, pose, factor=1):
            for iperson, personjoint in enumerate(pose):
                color = np.random.randint(0, 256, size=3, dtype=np.uint8).tolist()
                if torch.is_tensor(personjoint):
                    personjoint = personjoint.numpy()
                personjoint[:, :2] = personjoint[:, :2] * factor
                personjoint = personjoint.astype(int)

                for ijoint, joint in enumerate(personjoint):
                    if joint[2] > 0:
                        cv2.circle(img, tuple(joint[:2].tolist()), 3, color, thickness=-1)
                for iconnect, connect in enumerate(datasets.mscoco.PART_CONNECT):
                    joint1 = personjoint[connect[0]]
                    joint2 = personjoint[connect[1]]
                    if joint1[2] > 0 and joint2[2] > 0:
                        cv2.line(img, tuple(joint1[:2].tolist()), tuple(joint2[:2].tolist()), color, thickness=1)

        import matplotlib.pyplot as plt
        import cv2
        result = self.generate_output(transform_back=False)
        preds = result["pred"]
        scores = result["score"]
        image_ids = result["image_id"]
        gts = result["keypoint"]
        img = self.exp.val_dataset.restore_image(result["img"][0])
        # gt, pred
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(40, 20), gridspec_kw={"wspace": 0, "hspace": 0, "left": 0, "top": 1, "right": 1, "bottom": 0})
        for i, (pred, gt) in enumerate(zip(preds, gts)):
            draw_img_gt = img[i].copy()
            draw_img_pred = draw_img_gt.copy()
            draw_pose(draw_img_gt, gt, 4)
            draw_pose(draw_img_pred, pred, 4)
            axes[0].imshow(draw_img_gt)
            axes[0].axis("off")
            axes[1].imshow(draw_img_pred)
            axes[1].axis("off")
            break
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
