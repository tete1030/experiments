import torch
import torch.nn as nn
from torch.nn import DataParallel
import numpy as np
import cv2
import re
import os

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from experiments.baseexperiment import BaseExperiment, EpochContext

import pose.models as models
import pose.datasets as datasets
from pose.models.bayproj import AutoCorrProj
from pose.models.common import StrictNaNReLU
from pose.models.displacechan import DisplaceChannel
from pose.utils.transforms import fliplr_pts
from utils.globals import config, hparams, globalvars
from utils.log import log_i
from utils.train import adjust_learning_rate
from utils.checkpoint import save_pred

from utils.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from utils.checkpoint import load_pretrained_loose
from utils.lambdalayer import Lambda
import math

FACTOR = 4
NUM_PARTS = datasets.mscoco.NUM_PARTS

class GroupNormWrapper(nn.GroupNorm):
    def __init__(self, num_features, eps=1e-5, num_groups=32):
        assert num_features % num_groups == 0, "num_features({}) is not dividend by num_groups({})".format(num_features, num_groups)
        super(GroupNormWrapper, self).__init__(num_groups, num_features, eps=1e-5)

BatchNorm2dImpl = GroupNormWrapper

class Experiment(BaseExperiment):
    def init(self):
        self.num_parts = NUM_PARTS
        pretrained = hparams["model"]["resnet_pretrained"]
        if config.resume is not None:
            pretrained = None

        self.model = nn.DataParallel(BayMini(hparams["model"]["out_shape"][::-1], self.num_parts, pretrained=pretrained).cuda())
        self.optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, self.model.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams['weight_decay'])

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

    def epoch_start(self, epoch, step):
        self.cur_lr = adjust_learning_rate(self.optimizer, epoch, hparams["learning_rate"], hparams["schedule"], hparams["lr_gamma"])

    def iter_step(self, epoch_ctx:EpochContext, loss:torch.Tensor, progress:dict):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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

        det_map_gt_cuda = [dm.cuda() for dm in det_maps_gt]
        # dirty trick for debug
        if config.vis:
            globalvars.cur_img = img
        output_maps = self.model(img)
        # dirty trick for debug, release
        if config.vis:
            globalvars.cur_img = None

        loss = 0.
        for ilabel, (outv, gtv) in enumerate(zip(output_maps, det_map_gt_cuda)):
            # if ilabel < len(det_map_gt_cuda) - 1:
            #     gtv *= (keypoint[:, :, 2] > 1.1).float().view(-1, self.num_parts, 1, 1).cuda()
            if ilabel < len(det_map_gt_cuda) - 1:
                loss = loss + ((outv - gtv).pow(2) * \
                    (keypoint[:, :, 2] != 1).float().view(-1, self.num_parts, 1, 1).cuda()).mean().sqrt()
            else:
                loss = loss + (outv - gtv).pow(2).mean().sqrt()

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


class BayMini(nn.Module):
    def __init__(self, output_shape, num_points, pretrained=None):
        super(BayMini, self).__init__()
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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BlockBase(nn.Module):
    def __init__(self):
        super(BlockBase, self).__init__()
    
    def _init_extra_mod(self):
        width = hparams["model"]["inp_shape"][0] // self.inshape_factor
        height = hparams["model"]["inp_shape"][1] // self.inshape_factor
        if not hparams["model"]["detail"]["use_dconv"]:
            displace = DisplaceChannel(height, width,
                                       hparams["model"]["detail"]["displace_stride"][self.res_index],
                                       fill=hparams["model"]["detail"]["displace_fill"],
                                       learnable_offset=False,
                                       disable_displace=hparams["model"]["detail"]["disable_displace"],
                                       random_offset=hparams["model"]["detail"]["random_offset"],
                                       use_origin=hparams["model"]["detail"]["use_origin"],
                                       actual_stride=hparams["model"]["detail"]["actual_stride"][self.res_index],
                                       displace_size=hparams["model"]["detail"]["displace_size"][self.res_index])
            num_pos = displace.num_init_pos
            num_y = displace.num_init_y
            num_x = displace.num_init_x
            offset_channels = hparams["model"]["detail"]["channels_per_pos"][self.res_index] * displace.num_init_pos
            displace = nn.Sequential(
                displace,
                nn.Conv2d(offset_channels, self.inplanes // 4, kernel_size=1, stride=1))
        else:
            assert not hparams["model"]["detail"]["disable_displace"]
            assert not hparams["model"]["detail"]["displace_fill"]
            assert not hparams["model"]["detail"]["random_offset"]
            assert hparams["model"]["detail"]["use_origin"]
            num_y, num_x, num_pos = DisplaceChannel.get_num_offset(
                height,
                width,
                hparams["model"]["detail"]["displace_size"][self.res_index],
                hparams["model"]["detail"]["displace_stride"][self.res_index],
                hparams["model"]["detail"]["displace_fill"],
                hparams["model"]["detail"]["use_origin"])
            offset_channels = hparams["model"]["detail"]["channels_per_pos"][self.res_index] * num_pos
            actual_stride = hparams["model"]["detail"]["actual_stride"][self.res_index] \
                                if hparams["model"]["detail"]["actual_stride"][self.res_index] is not None else \
                                hparams["model"]["detail"]["displace_stride"][self.res_index]
            displace = nn.Conv2d(offset_channels, self.inplanes // 4,
                                 kernel_size=(num_y, num_x),
                                 padding=(num_y // 2 * actual_stride, num_x // 2 * actual_stride),
                                 stride=1,
                                 dilation=actual_stride)
        print("Displace{}_{} configuration:".format(self.res_index, self.block_index))
        print("\tinp_channels=" + str(self.inplanes))
        print("\toffset_channels=" + str(offset_channels))
        print("\toffset_nx=" + str(num_x))
        print("\toffset_ny=" + str(num_y))
        return nn.Sequential(nn.Conv2d(self.inplanes, self.inplanes // 4, kernel_size=1, stride=1),
                             BatchNorm2dImpl(self.inplanes // 4),
                             StrictNaNReLU(inplace=True),
                             nn.Conv2d(self.inplanes // 4, self.inplanes // 4, kernel_size=3, stride=1, padding=1),
                             BatchNorm2dImpl(self.inplanes // 4),
                             StrictNaNReLU(inplace=True),
                             nn.Conv2d(self.inplanes // 4, offset_channels, kernel_size=1, stride=1),
                             BatchNorm2dImpl(offset_channels, num_groups=num_pos),
                             StrictNaNReLU(inplace=True),
                             displace,
                             StrictNaNReLU(inplace=True),
                             nn.Conv2d(self.inplanes // 4, self.inplanes // 4, kernel_size=3, stride=1, padding=1),
                             BatchNorm2dImpl(self.inplanes // 4),
                             StrictNaNReLU(inplace=True),
                             nn.Conv2d(self.inplanes // 4, self.inplanes, kernel_size=1, stride=1),
                             BatchNorm2dImpl(self.inplanes),
                             StrictNaNReLU(inplace=True))

    def _forward_extra_mod(self, x):
        if self.extra_mod is not None:
            extra_out = self.extra_mod(x)

            if config.vis and False:
                import matplotlib.pyplot as plt
                import cv2
                fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
                
                for row, axes_row in enumerate(axes):
                    img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                    fts = x.data[row].cpu().numpy()
                    for col, ax in enumerate(axes_row):
                        if col == 0:
                            ax.imshow(img)
                        else:
                            ax.imshow(fts[col-1])
                fig.suptitle("BasicBlock x")

                fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
                for row, axes_row in enumerate(axes):
                    img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                    fts = extra_out.data[row].cpu().numpy()
                    for col, ax in enumerate(axes_row):
                        if col == 0:
                            ax.imshow(img)
                        else:
                            ax.imshow(fts[col-1])
                fig.suptitle("BasicBlock extra_out")
                plt.show()

            return extra_out
        else:
            return None

class Bottleneck(BlockBase):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        if block_index == hparams["model"]["detail"]["block_index"][res_index]:
            self.use_extra_mod = True
        else:
            self.use_extra_mod = False
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = BatchNorm2dImpl(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2dImpl(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2dImpl(planes * 4)
        if self.use_extra_mod:
            self.relu = StrictNaNReLU(inplace=False)
        else:
            self.relu = StrictNaNReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index
        if self.use_extra_mod:
            self.extra_mod = self._init_extra_mod()
        else:
            self.extra_mod = None

    def forward(self, x):
        extra_out = self._forward_extra_mod(x)

        if extra_out is not None:
            x = x + extra_out

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

        return out

class BasicBlock(BlockBase):
    expansion = 1

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if block_index == hparams["model"]["detail"]["block_index"][res_index]:
            self.use_extra_mod = True
        else:
            self.use_extra_mod = False
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dImpl(planes)
        if self.use_extra_mod:
            self.relu = StrictNaNReLU(inplace=False)
        else:
            self.relu = StrictNaNReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dImpl(planes)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index
        if self.use_extra_mod:
            self.extra_mod = self._init_extra_mod()
        else:
            self.extra_mod = None

    def forward(self, x):
        extra_out = self._forward_extra_mod(x)

        if extra_out is not None:
            x = x + extra_out

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

        return out

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
            # if re.match(r"^(.+\.)?extra_mod(\..+)?$", mod_name):
            #     continue
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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
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
