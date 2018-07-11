#!python3
import torch
import numpy as np
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import MSELoss
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import math
import torch.utils.model_zoo as model_zoo

import pose.models as models
import pose.datasets as datasets
import pose.utils.config as config
from pose.utils.transforms import fliplr_pts

from pose.utils.misc import adjust_learning_rate

import cv2

FACTOR = 4

class Experiment(object):
    def __init__(self, hparams):
        self.num_parts = datasets.mscoco.NUM_PARTS
        self.hparams = hparams
        use_pretrained = (config.resume is not None)
        self.model = DataParallel(BayBaseline(hparams["model"]["out_shape"][::-1], self.num_parts, pretrained=use_pretrained).cuda())
        self.criterion = MSELoss().cuda()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                          lr=hparams['learning_rate'],
                                          weight_decay=hparams['weight_decay'])

        self.coco = COCO("data/mscoco/person_keypoints_train2014.json")

        self.train_dataset = datasets.COCOSinglePose("data/mscoco/images",
                                               self.coco,
                                               "data/mscoco/sp_split_3.pth",
                                               "data/mscoco/mean_std_3.pth",
                                               True,
                                               img_res=self.hparams["model"]["inp_shape"],
                                               ext_border=self.hparams["dataset"]["ext_border"],
                                               kpmap_res=self.hparams["model"]["out_shape"],
                                               keypoint_res=self.hparams["model"]["out_shape"],
                                               kpmap_sigma=self.hparams["model"]["gaussian_kernels"],
                                               scale_factor=self.hparams["dataset"]["scale_factor"],
                                               rot_factor=self.hparams["dataset"]["rotate_factor"],
                                               trans_factor=self.hparams["dataset"]["translation_factor"])

        self.val_dataset = datasets.COCOSinglePose("data/mscoco/images",
                                             self.coco,
                                             "data/mscoco/sp_split_3.pth",
                                             "data/mscoco/mean_std_3.pth",
                                             False,
                                             img_res=self.hparams["model"]["inp_shape"],
                                             ext_border=self.hparams["dataset"]["ext_border"],
                                             kpmap_res=self.hparams["model"]["out_shape"],
                                             keypoint_res=self.hparams["model"]["out_shape"],
                                             kpmap_sigma=self.hparams["model"]["gaussian_kernels"],
                                             scale_factor=self.hparams["dataset"]["scale_factor"],
                                             rot_factor=self.hparams["dataset"]["rotate_factor"],
                                             trans_factor=self.hparams["dataset"]["translation_factor"])

        self.train_collate_fn = datasets.COCOSinglePose.collate_function
        self.test_collate_fn = datasets.COCOSinglePose.collate_function
        self.worker_init_fn = datasets.mscoco.worker_init

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

    def epoch(self, epoch):
        self.hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, self.hparams['learning_rate'], self.hparams['schedule'], self.hparams['lr_gamma'])

    def process(self, batch, train, detail=None):
        image_ids = batch["img_index"].tolist()
        img = batch["img"]
        det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        img_ori_size = batch["img_ori_size"]
        keypoint = batch["keypoint"]
        batch_size = img.size(0)

        det_map_gt_vars = [dm.cuda() for dm in det_maps_gt]
        output_vars = self.model(img)

        loss = 0.
        for ilabel, (outv, gtv) in enumerate(zip(output_vars, det_map_gt_vars)):
            # if ilabel < len(det_map_gt_vars) - 1:
            #     gtv *= (keypoint[:, :, 2] > 1.1).float().view(-1, self.num_parts, 1, 1).cuda()
            loss += self.criterion(outv, gtv) / self.hparams["model"]["gaussian_kernels"][ilabel]

        if (loss.data != loss.data).any():
            import pdb; pdb.set_trace()

        if not train or config.vis:
            pred, score = parse_map(output_vars[-1])
            pred_affined = pred.copy()
            for samp_i in range(batch_size):
                pred_affined[samp_i, :, :2] = kpt_affine(pred_affined[samp_i, :, :2] * FACTOR, np.linalg.pinv(transform_mat[samp_i])[:2])
                if img_flipped[samp_i]:
                    pred_affined[samp_i] = fliplr_pts(pred_affined[samp_i], datasets.mscoco.FLIP_INDEX, width=img_ori_size[isamp, 0])
            ans = generate_ans(image_ids, pred_affined, score)
        else:
            pred = None
            ans = None

        if config.vis:
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
                        pt = pred[i, j]
                        if pt[2] > 0:
                            cv2.circle(draw_img, (int(pt[0] * FACTOR), int(pt[1] * FACTOR)), radius=2, color=(0, 0, 255), thickness=-1)
                    axes.flat[i].imshow(draw_img[..., ::-1])

                plt.show()

            if True:
                pred_resized = batch_resize((output_vars[-1].data.cpu().numpy().clip(0, 1) * 255).round().astype(np.uint8) , img.size()[-2:])
                nrows = 3; ncols = 6
                for i in range(batch_size):
                    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
                    for ax in axes.flat:
                        ax.axis("off")
                    for j in range(self.num_parts):
                        ax = axes.flat[j]
                        draw_img = cv2.addWeighted(img_restored[i], 1, cv2.applyColorMap(pred_resized[i, j, :, :, None], cv2.COLORMAP_HOT), 0.5, 0)
                        ax.imshow(draw_img[..., ::-1])
                        ax.set_title(datasets.mscoco.PART_LABELS[j])
                    plt.show()

        phase_str = "train" if train else "valid"
        config.tb_writer.add_scalars(config.exp_name + "/loss", {phase_str: loss.item()}, detail["step"])
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

class BayBaseline(nn.Module):
    def __init__(self, output_shape, num_points, pretrained=True):
        super(BayBaseline, self).__init__()
        self.resnet50 = resnet50(pretrained=pretrained)
        self.global_net = globalNet([2048, 1024, 512, 256], output_shape, num_points)

    def forward(self, x):
        res_out = self.resnet50(x)
        global_re, global_out = self.global_net(res_out)
        return global_out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x):
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

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = torch.load("pretrained/resnet50-19c8e357.pth")
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        from collections import OrderedDict
        state_dict = model.state_dict()
        pretrained_state_dict = torch.load("pretrained/resnet101-19c8e357.pth")
        for k, v in pretrained_state_dict.items():
            if k not in state_dict:
                continue
            state_dict[k] = v
        model.load_state_dict(state_dict)
    return model

class globalNet(nn.Module):
    def __init__(self, input_sizes, output_shape, num_points):
        super(globalNet, self).__init__()

        self.layer1_1 = self._make_layer1(input_sizes[0])
        self.layer1_2 = self._make_layer2()
        self.layer1_3 = self._make_layer3(output_shape, num_points)

        self.layer2_1 = self._make_layer1(input_sizes[1])
        self.layer2_2 = self._make_layer2()
        self.layer2_3 = self._make_layer3(output_shape, num_points)

        self.layer3_1 = self._make_layer1(input_sizes[2])
        self.layer3_2 = self._make_layer2()
        self.layer3_3 = self._make_layer3(output_shape, num_points)

        self.layer4_1 = self._make_layer1(input_sizes[3])
        self.layer4_3 = self._make_layer3(output_shape, num_points)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer1(self, input_size):

        layers = []

        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_layer2(self):

        layers = []

        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=True))

        return nn.Sequential(*layers)

    def _make_layer3(self, output_shape, num_points):

        layers = []

        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_points,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(num_points))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))

        return nn.Sequential(*layers)

    def forward(self, x):

        x1_1 = self.layer1_1(x[0])
        x1_2 = self.layer1_2(x1_1)
        x1_3 = self.layer1_3(x1_1)

        x2_1 = self.layer2_1(x[1]) + x1_2
        x2_2 = self.layer2_2(x2_1)
        x2_3 = self.layer2_3(x2_1)

        x3_1 = self.layer3_1(x[2]) + x2_2
        x3_2 = self.layer3_2(x3_1)
        x3_3 = self.layer3_3(x3_1)

        x4_1 = self.layer4_1(x[3]) + x3_2
        x4_3 = self.layer4_3(x4_1)

        return [x4_1, x3_1, x2_1, x1_1], [x4_3, x3_3, x2_3, x1_3]

def parse_map(det_map, thres=0.1, factor=4):
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

if __name__ == "__main__":
    def test_main():
        from ruamel.yaml import YAML
        import pose.utils.config as config
        import importlib

        exp_name = "baybase"

        with open('experiments/config.yaml', 'r') as f:
            conf = YAML(typ='safe').load(f)
            conf_data = conf["default"]
            config.__dict__.update(conf_data.items())

        config.exp_name = exp_name
        with open("experiments/hparams.yaml", "r") as f:
            hparams = YAML(typ="safe").load(f)[exp_name]

        config.checkpoint = config.checkpoint.format(**{'exp': exp_name, 'id': hparams['id']})
        if config.resume is not None:
            config.resume = config.resume.format(**{'exp': exp_name, 'id': hparams['id']})

        exp_module = importlib.import_module('experiments.' + exp_name)
        exp = exp_module.Experiment(hparams)

        exp.train_dataset.debug = True

        train_loader = torch.utils.data.DataLoader(
            exp.train_dataset,
            collate_fn=exp.train_collate_fn,
            batch_size=exp.hparams['train_batch'],
            num_workers=0,
            shuffle=True,
            pin_memory=True,
            drop_last=exp.train_drop_last if hasattr(exp, "train_drop_last") else False)

        for batch in train_loader:
            continue

    test_main()
