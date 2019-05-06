import torch
from torch import nn
import re
import math

from utils.checkpoint import save_pred, load_pretrained_loose
from utils.globals import config, hparams, globalvars
from .offset import OffsetBlock

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = globalvars.BatchNorm2dImpl(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = globalvars.BatchNorm2dImpl(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = globalvars.BatchNorm2dImpl(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if not (self.res_index in [1, 2, 3] and self.block_index == 1) and (self.res_index != 2 or self.block_index < 6) and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            expand_chan_ratio = hparams.MODEL.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO[OffsetBlock._counter]
            use_special = bool(hparams.MODEL.LEARNABLE_OFFSET.USE_SPECIAL[OffsetBlock._counter] > 0)
            if expand_chan_ratio > 0:
                self.offset_block = OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                    hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                    self.inplanes,
                    self.inplanes,
                    int(self.inplanes * expand_chan_ratio),
                    use_atten=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.ENABLE,
                    use_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SPACE_NORM,
                    use_post_atten=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.ENABLE,
                    use_post_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.SPACE_NORM,
                    use_transformer=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE and use_special,
                    use_arc=hparams.MODEL.LEARNABLE_OFFSET.ARC.ENABLE and use_special,
                    dpool_size=hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE if use_special else 0,
                    always_train_block=hparams.TRAIN.OFFSET.ALWAYS_TRAIN_BLOCK,
                    use_fusion=use_special)
                if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                    globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))
            else:
                self.offset_block = None
            OffsetBlock._counter += 1
        else:
            self.offset_block = None

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.pre_early_predictor_outs[x.device].append(x)

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, inshape_factor, res_index, block_index, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = globalvars.BatchNorm2dImpl(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = globalvars.BatchNorm2dImpl(planes)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor
        self.res_index = res_index
        self.block_index = block_index

        if not (self.res_index in [1, 2, 3] and self.block_index == 1) and (self.res_index != 2 or self.block_index < 6) and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            expand_chan_ratio = hparams.MODEL.LEARNABLE_OFFSET.EXPAND_CHAN_RATIO[OffsetBlock._counter]
            use_special = bool(hparams.MODEL.LEARNABLE_OFFSET.USE_SPECIAL[OffsetBlock._counter] > 0)
            if expand_chan_ratio > 0:
                self.offset_block = OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                    hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                    self.inplanes,
                    self.inplanes,
                    int(self.inplanes * expand_chan_ratio),
                    use_atten=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.ENABLE,
                    use_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SPACE_NORM,
                    use_post_atten=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.ENABLE,
                    use_post_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.SPACE_NORM,
                    use_transformer=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE and use_special,
                    use_arc=hparams.MODEL.LEARNABLE_OFFSET.ARC.ENABLE and use_special,
                    dpool_size=hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE if use_special else 0,
                    always_train_block=hparams.TRAIN.OFFSET.ALWAYS_TRAIN_BLOCK,
                    use_fusion=use_special)
                if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                    globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))
            else:
                self.offset_block = None
            OffsetBlock._counter += 1
        else:
            self.offset_block = None

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.pre_early_predictor_outs[x.device].append(x)

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
        self.bn1 = globalvars.BatchNorm2dImpl(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inshape_factor *= 2
        self.layer1 = self._make_layer(block, 64, layers[0], res_index=0)
        if not hparams.MODEL.DETAIL.FIRST_ESP_ONLY:
            self.layer2 = self._make_layer(block, 128, layers[1], res_index=1, stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], res_index=2, stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], res_index=3, stride=2)

        for mod_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, globalvars.BatchNorm2dImpl):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, res_index, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                globalvars.BatchNorm2dImpl(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=0, stride=stride, downsample=downsample))
        if stride != 1:
            self.inshape_factor *= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, inshape_factor=self.inshape_factor, res_index=res_index, block_index=i))

        self.early_predictor_source = bool(hparams.MODEL.DETAIL.EARLY_PREDICTOR and not hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK)
        if self.early_predictor_source and res_index < 3:
            globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))

        # Use BreakableSequential to support middle break 
        return BreakableSequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1, x2, x3, x4 = None, None, None, None

        x1 = self.layer1(x)
        if self.early_predictor_source:
            globalvars.pre_early_predictor_outs[x.device].append(x1)
        if hparams.MODEL.DETAIL.FIRST_ESP_ONLY:
            return None

        if x1 is not None:
            x2 = self.layer2(x1)
            if self.early_predictor_source:
                globalvars.pre_early_predictor_outs[x.device].append(x2)
        if x2 is not None:
            x3 = self.layer3(x2)
            if self.early_predictor_source:
                globalvars.pre_early_predictor_outs[x.device].append(x3)
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

def resnet101(pretrained=None, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained is not None:
        print("Loading pretrained resnet101 ...")
        model_state_dict = model.state_dict()
        model_state_dict = load_pretrained_loose(model_state_dict, torch.load(pretrained))
        model.load_state_dict(model_state_dict)
    return model
