import torch
from torch import nn
import os
import re
import math

from utils.checkpoint import save_pred, load_pretrained_loose
from utils.globals import config, hparams, globalvars
from utils.log import log_e, log_i, log_w

from .offset import OffsetBlock

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class OffsetBlockWrapper(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, inshape_factor, expand_chan_ratio, stride=1, downsample=None):
        super(OffsetBlockWrapper, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor

        if expand_chan_ratio > 0 and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            self.offset_block = OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.planes,
                int(self.inplanes * expand_chan_ratio),
                downsample=downsample)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))
        else:
            log_w("Empty OffsetBlockWrapper")
            self.offset_block = None

    def forward(self, x):
        if self.offset_block is not None:
            x = self.offset_block(x)
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.pre_early_predictor_outs[x.device].append(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, inshape_factor, expand_chan_ratio, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.inplanes = inplanes
        self.bn1 = nn.BatchNorm2d(planes, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor

        if expand_chan_ratio > 0 and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            self.offset_block = OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.inplanes,
                int(self.inplanes * expand_chan_ratio))
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))
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

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, inshape_factor, expand_chan_ratio, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.inshape_factor = inshape_factor

        if expand_chan_ratio > 0 and hparams.MODEL.DETAIL.ENABLE_OFFSET_BLOCK:
            self.offset_block = OffsetBlock(
                hparams.MODEL.INP_SHAPE[1] // self.inshape_factor,
                hparams.MODEL.INP_SHAPE[0] // self.inshape_factor,
                self.inplanes,
                self.inplanes,
                int(self.inplanes * expand_chan_ratio))
            if hparams.MODEL.DETAIL.EARLY_PREDICTOR and hparams.MODEL.DETAIL.EARLY_PREDICTOR_FROM_OFFBLK:
                globalvars.early_predictor_size.append((self.inplanes, self.inshape_factor))
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

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, offset_expand, multi_scale_output=True, enable_early_predictor=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels, offset_expand)
        self.fuse_layers = self._make_fuse_layers()
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR and enable_early_predictor:
            self.predictor_fuse_layer = self._make_predictor_fuse_layer(256)
            globalvars.early_predictor_size.append((256, 4))
        else:
            self.predictor_fuse_layer = None
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, offset_expand,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=hparams.TRAIN.BN_MOMENTUM
                ),
            )
        # TODO: strong assumption
        inshape_factor = 4 * (2**branch_index)
        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                inshape_factor=inshape_factor,
                expand_chan_ratio=offset_expand[branch_index][0],
                stride=stride,
                downsample=downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index],
                    inshape_factor=inshape_factor,
                    expand_chan_ratio=offset_expand[branch_index][i],
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels, offset_expand):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels, offset_expand)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def _make_predictor_fuse_layer(self, out_channels):
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layer = []
        for j in range(num_branches):
            if j > 0:
                fuse_layer.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[j],
                            out_channels,
                            1, 1, 0, bias=False
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.Upsample(scale_factor=2**j, mode='bilinear'),
                        nn.Conv2d(
                            out_channels,
                            out_channels,
                            3, 1, 1, bias=False, groups=out_channels
                        ),
                        nn.BatchNorm2d(out_channels),
                    )
                )
            elif j == 0:
                fuse_layer.append(
                    nn.Sequential(
                        nn.Conv2d(
                            num_inchannels[j],
                            out_channels,
                            1, 1, 0, bias=False
                        ),
                        nn.BatchNorm2d(out_channels)
                    )
                )
            
        fuse_layer = nn.ModuleList(fuse_layer)

        return fuse_layer

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        
        if self.predictor_fuse_layer:
            y = 0
            for i in range(0, len(self.predictor_fuse_layer)):
                y = y + self.predictor_fuse_layer[i](x[i])
            globalvars.pre_early_predictor_outs[y.device].append(y)

        return x_fuse


blocks_dict = {
    'basic': BasicBlock,
    'bottleneck': Bottleneck,
    'offset': OffsetBlockWrapper
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, num_classes):
        self.inplanes = 64
        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=hparams.TRAIN.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        block = blocks_dict[hparams.MODEL.HRNET.STAGE1.BLOCK]
        self.layer1 = self._make_layer(block, 64, hparams.MODEL.HRNET.STAGE1.NUM_BLOCKS, offset_expands=hparams.MODEL.HRNET.STAGE1.OFFSET_EXPAND, force_expansion=4)
        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            globalvars.early_predictor_size.append((256, 4))

        self.stage2_cfg = hparams.MODEL.HRNET.STAGE2
        num_channels = self.stage2_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage2_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = hparams.MODEL.HRNET.STAGE3
        num_channels = self.stage3_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage3_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = hparams.MODEL.HRNET.STAGE4
        num_channels = self.stage4_cfg.NUM_CHANNELS
        block = blocks_dict[self.stage4_cfg.BLOCK]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False, disable_early_predictor=True)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=num_classes,
            kernel_size=hparams.MODEL.HRNET.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if hparams.MODEL.HRNET.FINAL_CONV_KERNEL == 3 else 0
        )

        self.pretrained_layers = hparams.MODEL.HRNET.PRETRAINED_LAYERS

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, offset_expands, stride=1, force_expansion=None):
        downsample = None
        if force_expansion:
            expansion = force_expansion
        else:
            expansion = block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * expansion, momentum=hparams.TRAIN.BN_MOMENTUM),
            )

        layers = []
        # TODO: strong assumption on inshape_factor
        layers.append(block(self.inplanes, planes * expansion, 4, offset_expands[0], stride=stride, downsample=downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, 4, offset_expands[i]))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True, disable_early_predictor=False):
        num_modules = layer_config.NUM_MODULES
        num_branches = layer_config.NUM_BRANCHES
        num_blocks = layer_config.NUM_BLOCKS
        num_channels = layer_config.NUM_CHANNELS
        block = blocks_dict[layer_config.BLOCK]
        fuse_method = layer_config.FUSE_METHOD
        offset_expand = layer_config.OFFSET_EXPAND

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            if not disable_early_predictor and i == num_modules - 1:
                enable_early_predictor = True
            else:
                enable_early_predictor = False

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    offset_expand[i],
                    multi_scale_output=reset_multi_scale_output,
                    enable_early_predictor=enable_early_predictor
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        if hparams.MODEL.DETAIL.EARLY_PREDICTOR:
            globalvars.pre_early_predictor_outs[x.device].append(x)

        x_list = []
        for i in range(self.stage2_cfg.NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg.NUM_BRANCHES):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg.NUM_BRANCHES):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x = self.final_layer(y_list[0])

        return x

    # TODO: for offblk and others
    def init_weights(self, pretrained=''):
        log_i('=> Initing weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            log_i('=> Loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            log_e('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

def get_pose_net(num_classes, pretrained):
    model = PoseHighResolutionNet(num_classes)

    if pretrained is not None:
        model.init_weights(pretrained)

    return model
