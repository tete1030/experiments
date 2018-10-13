import os
import re
import math
import time
import itertools
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn import DataParallel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import pose.models as models
import pose.datasets as datasets
from pose.models.bayproj import AutoCorrProj
from pose.models.common import StrictNaNReLU
from pose.models.displacechan import DisplaceChannel
from pose.utils.transforms import fliplr_pts
from pose.utils.evaluation import AverageMeter, CycleAverageMeter
from utils.globals import config, hparams, globalvars
from utils.log import log_i, log_w, log_progress
from utils.train import adjust_learning_rate, TrainContext, ValidContext
from utils.checkpoint import save_pred
from utils.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from utils.checkpoint import load_pretrained_loose, save_checkpoint, RejectLoadError
from utils.lambdalayer import Lambda
from experiments.baseexperiment import BaseExperiment, EpochContext

FACTOR = 4
NUM_PARTS = datasets.mscoco.NUM_PARTS

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, num_groups=32):
        assert num_features % num_groups == 0, "num_features({}) is not dividend by num_groups({})".format(num_features, num_groups)
        super(GroupNormWrapper, self).__init__(num_groups, num_features, eps=1e-5)

BatchNorm2dImpl = GroupNormWrapper

class Experiment(BaseExperiment):
    exp = None
    def init(self):
        if Experiment.exp is not None:
            log_w("Singleton experiment is already taken")
        Experiment.exp = self
        self.early_predictor_size = list()
        self.displace_mods = list()
        if hparams["model"]["detail"]["early_predictor"]:
            self.early_predictors = list()
            self.pre_early_predictor_outs = dict()

        self.num_parts = NUM_PARTS
        pretrained = hparams["model"]["resnet_pretrained"]
        if config.resume is not None:
            pretrained = None

        self.model = nn.DataParallel(Controller(MainModel(hparams["model"]["out_shape"][::-1], self.num_parts, pretrained=pretrained).cuda()))

        self.offset_parameters = list(filter(lambda x: x.requires_grad, [dm.offset for dm in self.displace_mods] + list(itertools.chain.from_iterable([dm.offset_regressor.parameters() for dm in self.displace_mods]))))

        if hparams["model"]["detail"]["early_predictor"]:
            self.early_predictor_parameters = list(filter(lambda x: x.requires_grad, itertools.chain.from_iterable([ep.parameters() for ep in self.early_predictors])))
        else:
            self.early_predictor_parameters = []

        special_parameter_ids = list(map(lambda x: id(x), self.offset_parameters + self.early_predictor_parameters))
        self.normal_parameters = list(filter(lambda x: x.requires_grad and id(x) not in special_parameter_ids, self.model.parameters()))

        self.optimizer = torch.optim.Adam(
            self.normal_parameters,
            lr=hparams["learning_rate"],
            weight_decay=hparams['weight_decay'])

        offset_optimizer_args = [
            {"para_name": "offset_lr", "params": self.offset_parameters, "lr": hparams["learnable_offset"]["lr"], "init_lr": hparams["learnable_offset"]["lr"]}]

        self.offset_optimizer = torch.optim.Adam(offset_optimizer_args)

        if hparams["model"]["detail"]["early_predictor"]:
            self.early_predictor_optimizer = torch.optim.Adam(
                self.early_predictor_parameters,
                lr=hparams["learning_rate"],
                weight_decay=hparams['weight_decay'])
        else:
            self.early_predictor_optimizer = None

        self.criterion = nn.MSELoss()
        
        self.cur_lr = hparams["learning_rate"]

        self.coco = COCO("data/mscoco/person_keypoints_train2014.json")
        self.train_dataset = datasets.COCOSinglePose("data/mscoco/images",
                                               self.coco,
                                               "data/mscoco/sp_split.pth",
                                               "data/mscoco/" + hparams["dataset"]["mean_std_file"],
                                               True,
                                               img_res=hparams["model"]["inp_shape"],
                                               ext_border=hparams["dataset"]["ext_border"],
                                               kpmap_res=hparams["model"]["out_shape"],
                                               keypoint_res=hparams["model"]["out_shape"],
                                               kpmap_sigma=hparams["model"]["gaussian_kernels"],
                                               scale_factor=hparams["dataset"]["scale_factor"],
                                               rot_factor=hparams["dataset"]["rotate_factor"],
                                               trans_factor=hparams["dataset"]["translation_factor"])

        self.val_dataset = datasets.COCOSinglePose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/sp_split.pth",
                                             "data/mscoco/" + hparams["dataset"]["mean_std_file"],
                                             False,
                                             img_res=hparams["model"]["inp_shape"],
                                             ext_border=hparams["dataset"]["ext_border"],
                                             kpmap_res=hparams["model"]["out_shape"],
                                             keypoint_res=hparams["model"]["out_shape"],
                                             kpmap_sigma=hparams["model"]["gaussian_kernels"],
                                             scale_factor=hparams["dataset"]["scale_factor"],
                                             rot_factor=hparams["dataset"]["rotate_factor"],
                                             trans_factor=hparams["dataset"]["translation_factor"])
        
        self.train_collate_fn = datasets.COCOSinglePose.collate_function
        self.valid_collate_fn = datasets.COCOSinglePose.collate_function
        self.worker_init_fn = datasets.mscoco.worker_init
        self.print_iter_start = " | "

        self.move_dis_avgmeter = []
        for dm in self.displace_mods:
            self.move_dis_avgmeter.append(Experiment.OffsetCycleAverageMeter(hparams["learnable_offset"]["move_average_cycle"], (dm.offset.data * dm.scale).cpu()))

    def load_checkpoint(self, checkpoint_folder, checkpoint_file,
                        no_strict_model_load=False,
                        no_criterion_load=False,
                        no_optimizer_load=False):

        # Load checkpoint data
        checkpoint_full = os.path.join(checkpoint_folder, checkpoint_file)
        checkpoint = torch.load(checkpoint_full)
        if no_strict_model_load:
            model_state_dict = self.model.state_dict()
            try:
                model_state_dict = load_pretrained_loose(model_state_dict, checkpoint["state_dict"])
            except RejectLoadError:
                return None
            self.model.load_state_dict(model_state_dict)
        else:
            self.model.load_state_dict(checkpoint["state_dict"])
        if not no_criterion_load:
            self.criterion.load_state_dict(checkpoint["criterion"])
        if not no_optimizer_load:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.offset_optimizer.load_state_dict(checkpoint["offset_optimizer"])
            if self.early_predictor_optimizer:
                self.early_predictor_optimizer.load_state_dict(checkpoint["early_predictor_optimizer"])
        self.move_dis_avgmeter = checkpoint["move_dis_avgmeter"]
        return checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_folder, checkpoint_file, epoch):
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
            "offset_optimizer": self.offset_optimizer.state_dict(),
            "early_predictor_optimizer": self.early_predictor_optimizer.state_dict() if self.early_predictor_optimizer else None,
            "move_dis_avgmeter": self.move_dis_avgmeter
        }
        save_checkpoint(checkpoint_dict, checkpoint_folder=checkpoint_folder, checkpoint_file=checkpoint_file, force_replace=True)

    @staticmethod
    def _summarize_tensorboard(eval_result, params, step):
        def _summarize(ap, iou_thr=None, area_rng="all", max_dets=100, title=None):
            type_str = "AP" if ap==1 else "AR"
            if title is None:
                iou_str = "{:0.2f}-{:0.2f}".format(params.iouThrs[0], params.iouThrs[-1]) \
                    if iou_thr is None else "{:0.2f}".format(iou_thr)
                title = "{:<9}_{:>6s}_{:>3d}".format(iou_str, area_rng, max_dets)

            aind = [i for i, aRng in enumerate(params.areaRngLbl) if aRng == area_rng]
            mind = [i for i, mDet in enumerate(params.maxDets) if mDet == max_dets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = eval_result["precision"]
                # IoU
                if iou_thr is not None:
                    t = np.where(iou_thr == params.iouThrs)[0]
                    s = s[t]
                s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = eval_result["recall"]
                if iou_thr is not None:
                    t = np.where(iou_thr == params.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, type_str), {title: mean_s}, step)
            return mean_s

        _summarize(1, title="avg", max_dets=20)
        _summarize(1, title="i50", max_dets=20, iou_thr=.5)
        _summarize(1, title="i75", max_dets=20, iou_thr=.75)
        _summarize(1, title="med", max_dets=20, area_rng="medium")
        _summarize(1, title="lar", max_dets=20, area_rng="large")
        _summarize(0, title="avg", max_dets=20)
        _summarize(0, title="i50", max_dets=20, iou_thr=.5)
        _summarize(0, title="i75", max_dets=20, iou_thr=.75)
        _summarize(0, title="med", max_dets=20, area_rng="medium")
        _summarize(0, title="lar", max_dets=20, area_rng="large")

    def evaluate(self, epoch_ctx:EpochContext, epoch, step):
        if "annotates" not in epoch_ctx.stored:
            return
        annotates = epoch_ctx.stored["annotates"]
        image_ids = annotates["image_index"]
        ans = annotates["annotate"]
        if ans is not None and len(ans) > 0:
            coco_dets = self.coco.loadRes(ans)
            coco_eval = COCOeval(self.coco, coco_dets, "keypoints")
            coco_eval.params.imgIds = list(image_ids)
            coco_eval.params.catIds = [1]
            coco_eval.evaluate()
            coco_eval.accumulate()

            self._summarize_tensorboard(coco_eval.eval, coco_eval.params, step)
            coco_eval.summarize()
        else:
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AP"), {"avg": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AP"), {"i50": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AP"), {"i75": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AP"), {"med": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AP"), {"lar": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AR"), {"avg": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AR"), {"i50": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AR"), {"i75": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AR"), {"med": 0}, step)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "AR"), {"lar": 0}, step)

            print("No points")

    def process_stored(self, epoch_ctx:EpochContext, epoch, step):
        if config.store:
            for store_key in epoch_ctx.stored:
                if epoch == 0:
                    pred_file = "{}_evaluate.npy".format(store_key)
                else:
                    pred_file = "{}_{}.npy".format(store_key, epoch)
                save_pred(epoch_ctx.stored[store_key], checkpoint_folder=config.checkpoint, pred_file=pred_file)

    def set_offset_learning_rate(self, epoch, step):
        if step >= hparams["learnable_offset"]["train_min_step"] and hparams["learnable_offset"]["lr_decay_step"] > 0 and hparams["learnable_offset"]["lr_gamma"] > 0:
            step_offset = max(0, step - hparams["learnable_offset"]["train_min_step"])
        else:
            step_offset = 0

        for param_group in self.offset_optimizer.param_groups:
            if step_offset > 0:
                cur_lr_offset = param_group["init_lr"] * (hparams["learnable_offset"]["lr_gamma"] ** (float(step_offset) / hparams["learnable_offset"]["lr_decay_step"]))
                log_i("Set {} to {:.5f}".format(param_group["para_name"], cur_lr_offset))
            else:
                cur_lr_offset = param_group["init_lr"]
            param_group["lr"] = cur_lr_offset

    def set_offset_learning_para(self, epoch, step):
        for dm in self.displace_mods:
            if dm.LO_interpolate_kernel_type == "gaussian" and dm.learnable_offset and dm.LO_active:
                if step >= hparams["learnable_offset"]["train_min_step"] and hparams["learnable_offset"]["interpolate_gaussian_sigma_decay_step"] > 0 and hparams["learnable_offset"]["interpolate_gaussian_sigma_decay_rate"] > 0:
                    step_offset = max(0, step - hparams["learnable_offset"]["train_min_step"])
                    LO_sigma_new = float(dm.LO_sigma_init) * (hparams["learnable_offset"]["interpolate_gaussian_sigma_decay_rate"] ** (float(step_offset) / hparams["learnable_offset"]["interpolate_gaussian_sigma_decay_step"]))
                    LO_kernel_size_new = int(LO_sigma_new * 3) * 2 + 1
                    dm.set_learnable_offset_para(LO_kernel_size_new, LO_sigma_new)

                if dm.LO_kernel_size == 1:
                    dm.switch_LO_state(False)

    def epoch_start(self, epoch, step):
        self.cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams["learning_rate"], hparams["schedule"], hparams["lr_gamma"])

        self.set_offset_learning_rate(epoch, step)
        self.set_offset_learning_para(epoch, step)

    class OffsetCycleAverageMeter(object):
        """Computes and stores the cycled average of offset"""
        def __init__(self, size, first):
            self.size = size
            self.reset()
            self.update(first)

        def reset(self):
            self._pool = list()
            self._pointer = 0
            self.count = 0
            # self.val = None
            self.avg = None

        def update(self, val, n=1):
            for i in range(n):
                if self.count >= self.size:
                    self._pool[self._pointer] = val
                    self._pointer = (self._pointer + 1) % self.size
                else:
                    self._pool.append(val)
                    self._pointer = (self._pointer + 1) % self.size
                    self.count += 1
            
            if self.count > 1:
                # self.val = val - self._pool[(self._pointer + self.count - 2) % self.count]
                self.avg = ((val - self._pool[self._pointer % self.count]) / (self.count - 1)).abs().mean()
            else:
                # self.val = None
                self.avg = None

    @staticmethod
    def set_para_require_grad(paras, requires_grad):
        for para in paras:
            para.requires_grad = requires_grad

    def save_offsets(self, step):
        offset_disabled = True
        for dm in self.displace_mods:
            if dm.LO_active:
                offset_disabled = False
        if not offset_disabled:
            torch.save([(dm.offset.detach() * dm.scale).cpu() for dm in self.displace_mods], os.path.join(config.checkpoint, "offset_{}.pth".format(step)))

    def epoch_end(self, epoch, step):
        self.save_offsets(step)

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        optimize_offset = False
        if progress["step"] >= hparams["learnable_offset"]["train_min_step"]:
            optimize_offset = True

        self.optimizer.zero_grad()
        if optimize_offset:
            self.offset_optimizer.zero_grad()
        if self.early_predictor_optimizer:
            self.early_predictor_optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if optimize_offset:
            self.offset_optimizer.step()
        if self.early_predictor_optimizer:
            self.early_predictor_optimizer.step()

        if optimize_offset:
            move_dises = list()
            for idm in range(len(self.displace_mods)):
                self.move_dis_avgmeter[idm].update((self.displace_mods[idm].offset.detach() * self.displace_mods[idm].scale).cpu())
                move_dises.append(self.move_dis_avgmeter[idm].avg)
            globalvars.tb_writer.add_scalars("{}/{}".format(globalvars.exp_name, "move_dis"), {"mod": np.mean(move_dises)}, progress["step"] + 1)

        if (progress["step"] + 1) % hparams["learnable_offset"]["offset_save_interval"] == 0:
            self.save_offsets(progress["step"] + 1)

    def iter_process(self, epoch_ctx: EpochContext, batch: dict, progress: dict) -> dict:
        image_ids = batch["img_index"].tolist()
        img = batch["img"]
        det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        img_ori_size = batch["img_ori_size"]
        keypoint = batch["keypoint"]
        is_train = progress["train"]
        batch_size = img.size(0)

        if progress["step"] == hparams["learnable_offset"]["train_min_step"] and progress["train"]:
            self.save_offsets(progress["step"])

        det_map_gt_cuda = [dm.cuda() for dm in det_maps_gt]
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img
        output_maps, early_predictor_outputs = self.model(img)
        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        loss = 0.
        for ilabel, (outv, gtv) in enumerate(zip(output_maps, det_map_gt_cuda)):
            # if ilabel < len(det_map_gt_cuda) - 1:
            #     gtv *= (keypoint[:, :, 2] > 1.1).float().view(-1, self.num_parts, 1, 1).cuda()
            if ilabel < len(det_map_gt_cuda) - 1 and not hparams["model"]["detail"]["loss_invisible"]:
                loss = loss + ((outv - gtv).pow(2) * \
                    (keypoint[:, :, 2] != 1).float().view(-1, self.num_parts, 1, 1).cuda()).mean().sqrt()
            else:
                loss = loss + (outv - gtv).pow(2).mean().sqrt()

        if hparams["model"]["detail"]["early_predictor"]:
            assert len(early_predictor_outputs) == len(hparams["model"]["detail"]["early_predictor_label_index"])
            for ilabel, outv in enumerate(early_predictor_outputs):
                if not hparams["model"]["detail"]["loss_invisible"]:
                    loss = loss + ((outv - det_map_gt_cuda[hparams["model"]["detail"]["early_predictor_label_index"][ilabel]]).pow(2) * \
                        (keypoint[:, :, 2] != 1).float().view(-1, self.num_parts, 1, 1).cuda()).mean().sqrt()
                else:
                    loss = loss + (outv - det_map_gt_cuda[hparams["model"]["detail"]["early_predictor_label_index"][ilabel]]).pow(2).mean().sqrt()

        epoch_ctx.add_scalar("loss", loss.item())

        if (loss.data != loss.data).any():
            import ipdb; ipdb.set_trace()

        if not is_train or config.vis:
            kp_pred, score = parse_map(output_maps[-1], thres=hparams["model"]["parse_threshold"])
            kp_pred_affined = kp_pred.copy()
            for samp_i in range(batch_size):
                kp_pred_affined[samp_i, :, :2] = kpt_affine(kp_pred_affined[samp_i, :, :2] * FACTOR, np.linalg.pinv(transform_mat[samp_i])[:2])
                if img_flipped[samp_i]:
                    kp_pred_affined[samp_i] = fliplr_pts(kp_pred_affined[samp_i], datasets.mscoco.FLIP_INDEX, width=img_ori_size[samp_i, 0].item())
            ans = generate_ans(image_ids, kp_pred_affined, score)
            epoch_ctx.add_store("annotates", {"image_index": image_ids, "annotate": ans})

            if config.store and hparams["config"]["store_map"] and is_train:
                if not hasattr(epoch_ctx, "store_counter"):
                    epoch_ctx.store_counter = 0
                if epoch_ctx.store_counter < 30:
                    epoch_ctx.add_store("pred", {"image_index": image_ids, "img": img, "gt": det_maps_gt, "pred": output_maps})
                epoch_ctx.store_counter += 1

        if config.vis and False:
            import matplotlib.pyplot as plt
            img_restored = np.ascontiguousarray(self.train_dataset.restore_image(img.data.cpu().numpy())[..., ::-1])

            if False:
                nrows = int(np.sqrt(float(batch_size)))
                ncols = (batch_size + nrows - 1) // nrows
                fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                for ax in axes.flat:
                    ax.axis("off")
                for i in range(batch_size):
                    draw_img = img_restored[i].copy()
                    for j in range(self.num_parts):
                        pt = kp_pred_affined[i, j]
                        if pt[2] > 0:
                            cv2.circle(draw_img, (int(pt[0] * FACTOR), int(pt[1] * FACTOR)), radius=2, color=(0, 0, 255), thickness=-1)
                    axes.flat[i].imshow(draw_img[..., ::-1])

            if False:
                for i in range(min(1, batch_size)):
                    nrows = 3; ncols = 6
                    for i_out in range(len(output_maps)):
                        pred_resized = batch_resize((output_maps[i_out][i].data.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8) , img.size()[-2:])
                        
                        fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                        fig.suptitle("%d" % (i_out,))
                        for ax in axes.flat:
                            ax.axis("off")
                        for j in range(self.num_parts):
                            ax = axes.flat[j]
                            draw_img = cv2.addWeighted(img_restored[i], 1, cv2.applyColorMap(pred_resized[j, :, :, None], cv2.COLORMAP_HOT), 0.5, 0)
                            ax.imshow(draw_img[..., ::-1])
                            ax.set_title(datasets.mscoco.PART_LABELS[j])
                    plt.show()

        result = {
            "loss": loss,
            "index": batch["index"]
        }

        return result

# Attention List:
# - middle_outputs
# - BreakableSequential in Blocks
# - 

class Controller(nn.Module):
    def __init__(self, main_model):
        super(Controller, self).__init__()
        self.main_model = main_model
        if hparams["model"]["detail"]["early_predictor"]:
            early_predictor = list()
            assert len(Experiment.exp.early_predictor_size) > 0
            for sz in Experiment.exp.early_predictor_size:
                early_predictor.append(Predictor(sz, hparams["model"]["out_shape"][::-1], NUM_PARTS))
            self.early_predictor = nn.ModuleList(early_predictor)
        else:
            self.early_predictor = None

    def forward(self, x):
        if self.early_predictor:
            Experiment.exp.pre_early_predictor_outs[x.device] = list()

        out = self.main_model(x)

        if self.early_predictor:
            pre_early_predictor_outs = Experiment.exp.pre_early_predictor_outs[x.device]
            Experiment.exp.pre_early_predictor_outs[x.device] = list()
            assert len(pre_early_predictor_outs) == len(self.early_predictor)
            return out, [self.early_predictor[i](pre_early_predictor_outs[i]) for i in range(len(pre_early_predictor_outs))]
        else:
            return out, None
        
class MainModel(nn.Module):
    def __init__(self, output_shape, num_points, pretrained=None):
        super(MainModel, self).__init__()
        if hparams["model"]["resnet"] == 18:
            self.resnet = resnet18(pretrained=pretrained)
            self.global_net = GlobalNet([512, 256, 128, 64], output_shape, num_points)
        elif hparams["model"]["resnet"] == 50:
            self.resnet = resnet50(pretrained=pretrained)
            self.global_net = GlobalNet([2048, 1024, 512, 256], output_shape, num_points)
        else:
            assert False

    def forward(self, x):
        res_out = self.resnet(x)
        global_re, global_out = self.global_net(res_out)
        return global_out

class Predictor(nn.Module):
    def __init__(self, inplanes, output_shape, num_class):
        super(Predictor, self).__init__()
        self.predict = self._make_predictor(inplanes, output_shape, num_class)
        Experiment.exp.early_predictors.append(self)
    
    def _make_predictor(self, inplanes, output_shape, num_class):
        layers = []
        # lateral of globalNet
        layers.append(nn.Conv2d(inplanes, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        # predict of globalNet
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.predict(x)

class SpaceSoftmax(nn.Module):
    def __init__(self):
        super(SpaceSoftmax, self).__init__()

    def forward(self, x):
        x_exp = x.exp()
        return x_exp / x_exp.sum(-1, keepdim=True).sum(-2, keepdim=True)

class OffsetBlock(nn.Module):
    def __init__(self, height, width, inplanes):
        super(OffsetBlock, self).__init__()
        LO_interpolate_kernel_type = hparams["learnable_offset"]["interpolate_kernel_type"]
        if LO_interpolate_kernel_type == "gaussian":
            LO_sigma = hparams["learnable_offset"]["interpolate_gaussian_sigma"]
            LO_kernel_size = int(LO_sigma * 3) * 2 + 1
        else:
            LO_sigma = 0.
            LO_kernel_size = 3
        self.inplanes = inplanes
        self.displace = DisplaceChannel(
            height, width,
            1,
            inplanes,
            learnable_offset=hparams["model"]["detail"]["displace_learnable_offset"],
            disable_displace=False,
            random_offset_init=hparams["model"]["detail"]["random_offset_init"],
            use_origin=True,
            actual_stride=1,
            displace_size=(1, 1),
            LO_interpolate_kernel_type=LO_interpolate_kernel_type,
            LO_kernel_size=LO_kernel_size,
            LO_sigma=LO_sigma,
            LO_balance_grad=False,
            free_chan_per_pos=inplanes // hparams["learnable_offset"]["bind_chan"],
            dconv_for_LO_stride=hparams["learnable_offset"]["dconv_for_LO_stride"],
            regress_offset=hparams["learnable_offset"]["regress_offset"],
            LO_half_reversed_offset=hparams["learnable_offset"]["half_reversed_offset"])
        Experiment.exp.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(inplanes, inplanes, 1)
        self.post_offset = nn.Conv2d(inplanes, inplanes, 1)
        self.atten = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            SpaceSoftmax())
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_pre = self.pre_offset(x)
        out_dis, out_dis_LO = self.displace(out_pre)
        if out_dis_LO is not None:
            out_dis = out_dis_LO
        out_atten = self.atten(x)
        out_post = self.post_offset(out_atten * out_dis)
        out_skip = x + out_post

        out_final = self.relu(self.bn(out_skip))

        if config.debug_nan:
            def get_backward_hook(var_name):
                def _backward_hook(grad):
                    exp = self
                    if isinstance(grad, torch.Tensor) and (grad.data != grad.data).any():
                        print("[OffsetBlock] " + var_name + " contains NaN during backward")
                        import ipdb; ipdb.set_trace()
                return _backward_hook

            all_var_names = ["x", "out_pre", "out_dis", "out_atten", "out_post", "out_skip", "out_final"]

            print("[OffsetBlock] !!!!!PERFORMANCE WARN: BACKWARD NAN DEBUGGING ENABLED!!!!!")
            for var_name in all_var_names:
                cur_var = locals()[var_name]
                if not (cur_var.data == cur_var.data).all():
                    print("[OffsetBlock] " + var_name + " contains NaN during forward")
                    import ipdb; ipdb.set_trace()
                cur_var.register_hook(get_backward_hook(var_name))

        return out_final

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = BatchNorm2dImpl(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2dImpl(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dImpl(planes * 4)
        self.relu = StrictNaNReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if stride == 1:
            self.offset_block = OffsetBlock(hparams["model"]["inp_shape"][1] // self.inshape_factor, hparams["model"]["inp_shape"][0] // self.inshape_factor, self.inplanes)
        else:
            self.offset_block = None

        if self.res_index in [0,1,2] and self.block_index == 1:
            self.early_prediction = True
        else:
            self.early_prediction = False

        if self.early_prediction:
            Experiment.exp.early_predictor_size.append(self.inplanes)

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        out = self.relu(out)

        if self.res_index in [0,1,2] and self.block_index == 1:
            Experiment.exp.pre_early_predictor_outs[out.device].append(out)

        return out

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dImpl(planes)
        self.relu = StrictNaNReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dImpl(planes)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if stride == 1:
            self.offset_block = OffsetBlock(hparams["model"]["inp_shape"][1] // self.inshape_factor, hparams["model"]["inp_shape"][0] // self.inshape_factor, self.inplanes)
        else:
            self.offset_block = None

        if self.res_index in [0,1,2] and self.block_index == 1:
            self.early_prediction = True
        else:
            self.early_prediction = False

        if self.early_prediction:
            Experiment.exp.early_predictor_size.append(self.inplanes)

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        if self.res_index in [0,1,2] and self.block_index == 1:
            Experiment.exp.pre_early_predictor_outs[out.device].append(out)

        return out

class BreakableSequential(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
            if input is None:
                return None
        return input

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        self.inshape_factor = 1
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.inshape_factor *= 2
        self.bn1 = BatchNorm2dImpl(64)
        self.relu = StrictNaNReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inshape_factor *= 2
        self.layer1 = self._make_layer(block, 64, layers[0], res_index=0)
        self.layer2 = self._make_layer(block, 128, layers[1], res_index=1, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], res_index=2, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], res_index=3, stride=2)

        for mod_name, m in self.named_modules():
            # TODO:
            if re.match(r"^(.+\.)?displace(\..+)?$", mod_name):
                continue
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, res_index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2dImpl(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=0, stride=stride, downsample=downsample))
        if stride != 1:
            self.inshape_factor *= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=i))

        # TODO:
        # Use BreakableSequential to support middle break 
        return BreakableSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # TODO:
        x1, x2, x3, x4 = None, None, None, None

        # TODO:
        x1 = self.layer1(x)
        if x1 is not None:
            x2 = self.layer2(x1)
        if x2 is not None:
            x3 = self.layer3(x2)
        if x3 is not None:
            x4 = self.layer4(x3)

        return [x4, x3, x2, x1]

def resnet18(pretrained=None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet18 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model

def resnet50(pretrained=None, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet50 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model

class GlobalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))
        layers.append(BatchNorm2dImpl(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(BatchNorm2dImpl(256))
        layers.append(StrictNaNReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(num_class, num_class,
            kernel_size=3, stride=1, groups=num_class, padding=1, bias=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs


def parse_map(det_map, thres=0.1):
    det_map = det_map.detach()
    if det_map.is_cuda:
        det_map = det_map.cpu()
    det_map = det_map.numpy()
    num_batch = det_map.shape[0]
    num_part = det_map.shape[1]
    height = det_map.shape[2]
    width = det_map.shape[3]

    pred = np.zeros((num_batch, num_part, 3), dtype=np.float32)
    score = np.zeros((num_batch, num_part), dtype=np.float32)
    for sample_i in range(num_batch):
        for part_i in range(num_part):
            loc = det_map[sample_i, part_i].argmax().item()
            y = loc // width
            x = loc % width
            score_sp = det_map[sample_i, part_i, y, x]
            # TODO: test always 1 and always store score
            if score_sp > thres:
                pred[sample_i, part_i, 2] = 1
                score[sample_i, part_i] = score_sp
            if det_map[sample_i, part_i, y, max(0, x-1)] < det_map[sample_i, part_i, y, min(width-1, x+1)]:
                off_x = 0.25
            else:
                off_x = -0.25
            if det_map[sample_i, part_i, max(0, y-1), x] < det_map[sample_i, part_i, min(height-1, y+1), x]:
                off_y = 0.25
            else:
                off_y = -0.25
            pred[sample_i, part_i, 0] = x + 0.5 + off_x
            pred[sample_i, part_i, 1] = y + 0.5 + off_y
    return pred, score

def generate_ans(image_ids, preds, scores):
    ans = []
    for sample_i in range(len(preds)):
        image_id = image_ids[sample_i]

        val = preds[sample_i]
        score = scores[sample_i].mean()
        if val[:, 2].max()>0:
            tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(score)}
            # # p: average detected locations
            # p = val[val[:, 2] > 0][:, :2].mean(axis = 0)
            # for j in val:
            #     if j[2]>0.:
            #         tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
            #     else:
            #         # TRICK: for not detected points, place them at the average point
            #         tmp["keypoints"] += [float(p[0]), float(p[1]), 0]
            tmp["keypoints"] = val.ravel().tolist()
            ans.append(tmp)
    return ans

def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)

def batch_resize(im, new_shape):
    assert isinstance(new_shape, tuple) and len(new_shape) == 2 and isinstance(new_shape[0], int) and isinstance(new_shape[1], int)
    im_pre_shape = im.shape[:-2]
    im_post_shape = im.shape[-2:]
    if im_post_shape == new_shape:
        return im
    im = im.reshape((-1,) + im_post_shape)
    return np.array([cv2.resize(im[i], (new_shape[1], new_shape[0])) for i in range(im.shape[0])]).reshape(im_pre_shape + new_shape)
