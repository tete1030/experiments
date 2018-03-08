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
            models.PoseNet(inp_dim=self.num_parts*2, merge_inp_dim=3, out_dim=self.num_parts*2,
                           hg_dim=self.hparams["model"]["hg_dim"],
                           bn=self.hparams["model"]["bn"]).cuda())

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

        self.train_pose_mgr = models.PoseManager(self.hparams["train_batch"],
                                                 self.num_parts,
                                                 (INP_RES, INP_RES),
                                                 self.hparams["model"]["max_person"],
                                                 cuda=True,
                                                 sigma=1*FACTOR)

        self.test_pose_mgr = models.PoseManager(self.hparams["test_batch"],
                                                self.num_parts,
                                                (INP_RES, INP_RES),
                                                self.hparams["model"]["max_person"],
                                                cuda=True,
                                                sigma=1*FACTOR)

        self.train_collate_fn = datasets.COCOPose.collate_function
        self.test_collate_fn = datasets.COCOPose.collate_function

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

    def summary_image(self, img, pred, gt, move_field, indices_TP, mask, title, step):
        """
        Draw summary

        - $tb_num images
        - 4 rows x 18 columns
            - first column: init_locate+move_1, all_keypoints_1+move_2, all_keypoints_2+move_3, all_keypoints_3+gt+match_line
            - remaining columns: keypoint_1+field_1+move_1, keypoint_2+field_2+move_2, keypoint_3+field_3+move_3, keypoint_4+gt+match_line
        """

        # DPI = 300
        TP_COLOR = (95, 162, 44)
        FP_COLOR = (0, 0, 255)
        CP_COLOR = (255, 0, 0)
        SEP_POINT_RADIUS = 3
        ALL_POINT_RADIUS = 2
        SEP_LINE_THICKNESS = 1
        ALL_LINE_THICKNESS = 1
        LEGEND_R = 20
        FORCE_MAX_LENGTH = float(INP_RES) / 2
        def generate_field_img(field_x, field_y):
            field_rho = np.sqrt(field_x**2 + field_y**2)
            field_phi = np.arctan2(field_y, field_x)
            fieldimg_h = (field_phi/np.pi + 1) / 2
            assert ((fieldimg_h >= 0) & (fieldimg_h <= 1)).all()
            fieldimg_s = np.ones(fieldimg_h.shape)
            fieldimg_v = field_rho.clip(max=FORCE_MAX_LENGTH) / FORCE_MAX_LENGTH
            fieldimg = mpl.colors.hsv_to_rgb(np.stack((fieldimg_h, fieldimg_s, fieldimg_v), axis=-1))
            return fieldimg

        tb_num = 6
        tb_img = img[:tb_num].numpy() + self.train_dataset.mean[None, :, None, None]
        nrows = self.hparams["model"]["stack"] + 1
        ncols = self.num_parts + 1
        whole_img = np.zeros((tb_num, nrows, ncols, 3, INP_RES, INP_RES))
        for i in range(len(tb_img)):
            img_trans = tb_img[i].transpose(1, 2, 0)

            for k in range(4):
                all_keypoints_img = (img_trans * 255).round().clip(0, 255).astype(np.uint8)
                for j in range(self.num_parts):
                    if k < len(move_field):
                        # Draw pred field
                        field_x_ori = move_field[k][i, j].numpy()
                        field_y_ori = move_field[k][i, j+self.num_parts].numpy()
                        field_x = ndimage.zoom(field_x_ori, FACTOR)
                        field_y = ndimage.zoom(field_y_ori, FACTOR)
                        fieldimg = generate_field_img(field_x, field_y)
                        finalimg = img_trans + 0.7 * fieldimg
                    else:
                        finalimg = img_trans

                    finalimg = (finalimg * 255).round().clip(0, 255).astype(np.uint8)

                    if k == 0:
                        cv2.putText(finalimg, datasets.mscoco.PART_LABELS[j], (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1)

                    pred_ki = pred[k][i].numpy()
                    if pred_ki.ndim > 2 and len(pred_ki) > 0:
                        pred_ki = pred_ki[:, j]
                    gt_i = gt[i].numpy()
                    if len(gt_i) > 0:
                        gt_i = gt_i[:, j]

                    for iper, (point_pred, point_gt) in enumerate(zip(pred_ki, gt_i)):
                        # Assume the pred and gt visible attribute has been updated
                        point_pred_visible = (len(point_pred) == 2 or point_pred[2] > 0)
                        point_gt_visible = (point_gt[2] > 0)
                        point_pred = tuple(point_pred[:2].astype(int).tolist())
                        point_gt = tuple(point_gt[:2].astype(int).tolist())
                        if k < len(move_field):
                            color_pred = FP_COLOR[::-1]
                            if point_pred_visible:
                                point_pred_in_field = (int(point_pred[0]/FACTOR), int(point_pred[1]/FACTOR))
                                force = (int(round(np.clip(field_x_ori[point_pred_in_field[1], point_pred_in_field[0]], -INP_RES, INP_RES))),
                                         int(round(np.clip(field_y_ori[point_pred_in_field[1], point_pred_in_field[0]], -INP_RES, INP_RES))))
                                point2_pred = (point_pred[0] + force[0], point_pred[1] + force[1])
                                # Draw force
                                cv2.arrowedLine(finalimg, point_pred, point2_pred, FP_COLOR[::-1], thickness=SEP_LINE_THICKNESS)
                                cv2.arrowedLine(all_keypoints_img, point_pred, point2_pred, FP_COLOR[::-1], thickness=ALL_LINE_THICKNESS)
                        else:
                            if (iper, j) in indices_TP[i]:
                                color_pred = TP_COLOR[::-1]
                                # Connect pred and gt
                                cv2.line(finalimg, point_pred, point_gt, TP_COLOR[::-1], thickness=SEP_LINE_THICKNESS)
                                cv2.line(all_keypoints_img, point_pred, point_gt, TP_COLOR[::-1], thickness=ALL_LINE_THICKNESS)
                            else:
                                color_pred = FP_COLOR[::-1]

                        if point_gt_visible:
                            # Draw gt
                            cv2.circle(finalimg, point_gt, SEP_POINT_RADIUS, CP_COLOR[::-1], thickness=-1)
                            cv2.circle(all_keypoints_img, point_gt, ALL_POINT_RADIUS, CP_COLOR[::-1], thickness=-1)

                        if point_pred_visible:
                            # Draw pred
                            cv2.circle(finalimg, point_pred, SEP_POINT_RADIUS, color_pred, thickness=-1)
                            cv2.circle(all_keypoints_img, point_pred, ALL_POINT_RADIUS, color_pred, thickness=-1)

                    whole_img[i, k, j+1] = finalimg.transpose(2, 0, 1).astype(np.float32) / 255
                whole_img[i, k, 0] = all_keypoints_img.transpose(2, 0, 1).astype(np.float32) / 255

        legend_X, legend_Y = np.meshgrid(np.arange(-LEGEND_R, LEGEND_R+1, dtype=np.float32), np.arange(-LEGEND_R, LEGEND_R+1, dtype=np.float32))
        legend_rho = np.sqrt(legend_X ** 2 + legend_Y ** 2)
        legend_phi = np.arctan2(legend_Y, legend_X)
        legend_hsv = np.zeros((LEGEND_R*2+1, LEGEND_R*2+1, 3), dtype=np.float32)
        legend_hsv[:, :, 0] = (legend_phi/np.pi + 1) / 2
        legend_hsv[:, :, 1] = 1
        legend_hsv[:, :, 2] = legend_rho / LEGEND_R
        legend_mask = (legend_rho <= LEGEND_R)
        legend = mpl.colors.hsv_to_rgb(legend_hsv).transpose(2, 0, 1)
        whole_img[0, 0, 0, :, (-LEGEND_R*2-1):, (-LEGEND_R*2-1):][:, legend_mask] = legend[:, legend_mask]

        whole_img = vutils.make_grid(torch.from_numpy(whole_img.reshape((-1, 3, INP_RES, INP_RES))),
                                     nrow=ncols, range=(0, 1), pad_value=0.3) # that the nrow equals to ncols is expected because make_grid use nrow as ncol
        config.tb_writer.add_image(title, whole_img, step)


    def summary_histogram(self, n_iter):
        for name, param in self.model.named_parameters():
            config.tb_writer.add_histogram("stretch." + name, param.clone().cpu().data.numpy(), n_iter, bins="doane")

    @profile
    def process(self, batch, train, detail=None):
        num_stack = self.hparams["model"]["stack"]
        init_noise_factor = self.hparams["model"]["init_noise_factor"]
        mid_noise_factor = self.hparams["model"]["mid_noise_factor"]
        use_outsider = self.hparams["model"]["use_outsider"]

        model_reg = self.model
        criterion_dis = self.criterion

        img = batch["img"]
        keypoint_gt = batch["keypoint"]
        locate_gt = batch["locate"]
        mask = batch["mask"]
        locate_std_gt = batch["locate_std"]
        locate_in_kp = batch["locate_in_kp"]

        volatile = not train

        img_var = torch.autograd.Variable(img.cuda(async=True), volatile=volatile)

        for i in range(len(locate_gt)):
            if locate_gt[i] is None:
                locate_gt[i] = torch.FloatTensor(0)

        for i in range(len(locate_std_gt)):
            if locate_std_gt[i] is not None:
                lsg_i = [float(std) if std is not None else (1.*FACTOR) for std in locate_std_gt[i]]
                locate_std_gt[i] = torch.FloatTensor(lsg_i)

        # Filter keypoints with locate
        # Change visible attribute of outsider to 0
        for bi in range(len(keypoint_gt)):
            lik_bi = locate_in_kp[bi]
            kp_bi = keypoint_gt[bi]
            if lik_bi is not None:
                kp_bi = kp_bi[lik_bi]
                if not use_outsider:
                    outsider_id = torch.nonzero(((kp_bi < 0) | (kp_bi >= INP_RES))[:, :, :2].sum(dim=-1))
                    if len(outsider_id) > 0:
                        kp_bi[outsider_id[:, 0],
                              outsider_id[:, 1],
                              torch.LongTensor([2]).expand(outsider_id.size(0))] = 0
            else:
                kp_bi = torch.FloatTensor(0)
            keypoint_gt[bi] = kp_bi

        locate_init = locate_gt
        locate_std_init = locate_std_gt

        if train:
            if init_noise_factor > 0:
                for ib in range(len(locate_init)):
                    num_person = len(locate_init[ib])
                    if num_person > 0:
                        locate_init_noise = torch.randn(num_person, 2).clamp_(-3, 3) * locate_std_init[ib].view(-1, 1).expand(num_person, 2) / float(init_noise_factor)
                        new_locate = locate_init[ib].float() + locate_init_noise
                        new_locate.round_().clamp_(0, INP_RES-1)
                    else:
                        new_locate = locate_init[ib]
                    locate_init[ib] = new_locate.long()

        if np.sum([len(loc_samp) for loc_samp in locate_init]) > 0:
            keypoint_pred_list = list()
            move_field_list = list()

            pose_mgr = self.train_pose_mgr if train else self.test_pose_mgr
            pose_mgr.init_with_locate(locate_init)
            if detail["summary"]:
                keypoint_pred_list.append(locate_init)

            model_reg(merge_inp=img_var)

            for i in range(num_stack):
                posemap = pose_mgr.generate()
                posemap_var = torch.autograd.Variable(posemap, volatile=volatile)
                move_field, _ = model_reg(inp=posemap_var, merge=True, free_merge=(i >= num_stack - 1))
                pose_mgr.move_keypoints(move_field, factor=FACTOR)
                if detail["summary"]:
                    move_field_list.append(move_field.data.cpu())
                    keypoint_pred_list.append(pose_mgr.get_split_keypoints())
                # TODO: IMPROVEMENT add noise perturb to keypoints to make model robust

            # keypoint_gt_cat: #all_match_person x #part x 3
            keypoint_gt_cat = torch.cat(keypoint_gt, dim=0)
            keypoint_gt_cat_mask = torch.autograd.Variable((keypoint_gt_cat[:, :, 2] > 0).cuda(), requires_grad=False)
            keypoint_gt_cat_var = torch.autograd.Variable(keypoint_gt_cat[:, :, :2].cuda(), requires_grad=False)

            keypoint_pred_cat_var = pose_mgr.all_keypoints_var
            loss = criterion_dis(keypoint_pred_cat_var, keypoint_gt_cat_var, keypoint_gt_cat_mask)

            keypoint_pred = pose_mgr.get_split_keypoints()

            precision, recall, _, _, indices_TP = PR_multi(keypoint_pred, keypoint_gt, num_parts=self.num_parts, threshold=float(INP_RES)/self.hparams["model"]["eval_threshold_factor"])

            if detail["summary"]:
                self.summary_image(img=img,
                                   pred=keypoint_pred_list,
                                   gt=keypoint_gt,
                                   move_field=move_field_list,
                                   indices_TP=indices_TP,
                                   mask=mask,
                                   title="stretch/" + ("train" if train else "val"),
                                   step=detail["epoch"] + 1)

        else:
            loss = None
            precision = None
            recall = None
            keypoint_pred = [torch.FloatTensor(0) for i in range(len(img))]

        result= {"loss": loss,
                 "acc": recall,
                 "recall": recall,
                 "prec": precision,
                 "index": batch["index"],
                 "pred": keypoint_pred}

        return result
