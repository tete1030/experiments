#!python3
import torch
import numpy as np
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import MSELoss
from pycocotools.coco import COCO

import math
import torch.utils.model_zoo as model_zoo

import pose.datasets as datasets
import pose.utils.config as config

from pose.utils.misc import adjust_learning_rate

class Experiment(object):
    def __init__(self, hparams):
        self.num_parts = datasets.mscoco.NUM_PARTS
        self.hparams = hparams
        use_pretrained = (config.resume is not None)
        self.model = DataParallel(BayBaseline(hparams["model"]["out_shape"], self.num_parts, pretrained=use_pretrained).cuda())
        self.criterion = MSELoss().cuda()
        self.optimizer = torch.optim.Adam(list(self.model.parameters()),
                                          lr=hparams['learning_rate'],
                                          weight_decay=hparams['weight_decay'])

        self.coco = COCO("data/mscoco/person_keypoints_train2014.json")

        self.train_dataset = datasets.COCOSinglePose("data/mscoco/images",
                                               self.coco,
                                               "data/mscoco/sp_split.pth",
                                               "data/mscoco/mean_std.pth",
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
                                             "data/mscoco/sp_split.pth",
                                             "data/mscoco/mean_std.pth",
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

    def epoch(self, epoch):
        self.hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, self.hparams['learning_rate'], self.hparams['schedule'], self.hparams['lr_gamma'])

    def process(self, batch, train, detail=None):
        img = batch["img"]
        det_maps_gt = batch["keypoint_map"]
        transform_mat = batch["img_transform"]
        img_flipped = batch["img_flipped"]
        keypoint = batch["keypoint"]
        volatile = not train

        det_map_gt_vars = [dm.cuda() for dm in det_maps_gt]
        output_vars = self.model(img)

        loss = 0.
        for ilabel, (outv, gtv) in enumerate(zip(output_vars, det_map_gt_vars)):
            if ilabel < len(det_map_gt_vars) - 1:
                gtv *= (keypoint[:, :, 2] > 1.1).float().view(-1, self.num_parts, 1, 1).cuda()
            loss += self.criterion(outv, gtv)
        loss = loss / len(det_map_gt_vars)

        if (loss.data != loss.data).any():
            import pdb; pdb.set_trace()

        phase_str = "train" if train else "valid"
        config.tb_writer.add_scalars(config.exp_name + "/loss", {phase_str: loss.data.cpu()[0]}, detail["step"])
        result = {
            "loss": loss,
            "acc": 0,
            "recall": 0,
            "prec": None,
            "index": batch["index"],
            "pred": None
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
