import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.globals import config, hparams, globalvars
from utils.log import log_i, log_w, log_progress
from lib.models.spacenorm import SpaceNormalization
from lib.models.displacechan import DisplaceChannel, PositionalGaussianDisplaceModule, OffsetTransformer

class Attention(nn.Module):
    def __init__(self, inplanes, outplanes, input_shape=None, bias_planes=0, bias_factor=0, space_norm=True, stride=1):
        super(Attention, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.total_inplanes = inplanes
        self.input_shape = input_shape
        self.bias_planes = bias_planes
        self.bias_factor = bias_factor
        if input_shape is not None and bias_planes > 0 and bias_factor > 0:
            self.total_inplanes += bias_planes
            bias_shape = (int(input_shape[0] // bias_factor), int(input_shape[1] // bias_factor))
            if config.check:
                log_i("bias_shape = {}".format(str(bias_shape)))
            self.bias = nn.Parameter(torch.ones(1, bias_planes, bias_shape[0], bias_shape[1], dtype=torch.float))
        else:
            self.bias = None

        self.space_norm = space_norm
        if space_norm:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
                nn.Softplus(),
                SpaceNormalization())
        else:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
                nn.Sigmoid())

    def forward(self, x):
        if self.bias is not None:
            atten_bias = nn.functional.interpolate(self.bias, size=x.size()[-2:], mode="bilinear", align_corners=True).expand(x.size(0), -1, -1, -1)
            if self.inplanes > 0:
                x = torch.cat([x, atten_bias], dim=1)
            else:
                x = atten_bias

        return self.atten(x)

class DynamicPooling(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(DynamicPooling, self).__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1
        x = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).view(1, -1).expand(kernel_size, -1)
        y = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1).view(-1, 1).expand(-1, kernel_size)
        dissq = torch.stack((x, y), dim=0).pow(2).float().sum(dim=0)
        self.register_buffer("dissq", dissq)
        self.sigma = nn.Parameter(torch.zeros(num_channels))
        self.sigma.data.fill_(kernel_size / 2 / 2)
        self.register_buffer("max_sigma", torch.tensor(kernel_size / 2, dtype=torch.float))
        self.eps = np.finfo(np.float32).eps.item()
        globalvars.dpools.append(self)

    def forward(self, x):
        kernel = torch.exp(-(self.dissq / 2)[None] / (self.sigma.pow(2)[:, None, None] + self.eps))
        kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
        kernel = kernel.view(self.num_channels, 1, self.kernel_size, self.kernel_size)
        expx = torch.exp(x.clamp(max=88.722835))
        gp_expx = F.conv2d(expx, kernel, padding=(self.kernel_size // 2, self.kernel_size // 2), groups=self.num_channels)
        pooled = torch.log(gp_expx + self.eps)
        return pooled

class OffsetBlock(nn.Module):
    _counter = 0
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1):
        super(OffsetBlock, self).__init__()

        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.displace_planes = displace_planes
        self.outplanes = outplanes
        self.stride = stride
        if config.check:
            log_i("inplanes={}, outplanes={}, in_height={}, in_width={}, out_height={}, out_width={}, displace_planes={}, num_offsets={}".format(
                self.inplanes,
                self.outplanes,
                self.height,
                self.width,
                self.out_height,
                self.out_width,
                self.displace_planes,
                self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN
            ))

        num_offset = self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE:
            offset_transformer = OffsetTransformer(
                self.inplanes,
                num_offset,
                num_regress=1,
                scale_grow_step=1 / hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None,
                absolute_regressor=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ABSOLUTE_REGRESSOR,
                sep_scale=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.SEP_SCALE)
        else:
            offset_transformer = None

        if hparams.MODEL.LEARNABLE_OFFSET.ARC.ENABLE:
            arc_displacer = PositionalGaussianDisplaceModule(
                num_offset,
                hparams.MODEL.LEARNABLE_OFFSET.ARC.NUM_SAMPLE,
                float(hparams.MODEL.LEARNABLE_OFFSET.ARC.ANGLE_STD) / 180 * np.pi,
                hparams.MODEL.LEARNABLE_OFFSET.ARC.SCALE_STD,
                min_angle_std=float(hparams.MODEL.LEARNABLE_OFFSET.ARC.MIN_ANGLE_STD) / 180 * np.pi,
                min_scale_std=hparams.MODEL.LEARNABLE_OFFSET.ARC.MIN_SCALE_STD,
                max_scale_std=5.1,
                sampler=hparams.MODEL.LEARNABLE_OFFSET.ARC.SAMPLER,
                weight_dist=hparams.MODEL.LEARNABLE_OFFSET.ARC.WEIGHT_DIST,
                learnable_sigma=False, transform_sigma=False,
                simple=hparams.MODEL.LEARNABLE_OFFSET.ARC.SIMPLE)
            globalvars.arc_displacers.append(arc_displacer)
        else:
            arc_displacer = None
        self.displace = DisplaceChannel(
            self.out_height, self.out_width,
            self.displace_planes, num_offset,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=hparams.MODEL.DETAIL.DISPLACE_LEARNABLE_OFFSET,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transformer=offset_transformer,
            half_reversed_offset=hparams.MODEL.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=globalvars.displace_mods[-1] if hparams.MODEL.LEARNABLE_OFFSET.REUSE_OFFSET and len(globalvars.displace_mods) > 0 else None,
            arc_gaussian=arc_displacer)
        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if hparams.MODEL.LEARNABLE_OFFSET.ATTEN.ENABLE:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SPACE_NORM, stride=stride)
        else:
            self.atten_displace = None
        if hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.ENABLE:
            self.atten_post = Attention(self.inplanes, self.outplanes, input_shape=(self.out_height, self.out_width), bias_planes=0, bias_factor=0, space_norm=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.SPACE_NORM, stride=stride)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

        if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1:
            self.dpool = DynamicPooling(self.displace_planes, hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE)
        else:
            self.dpool = None

    def forward(self, x):
        if config.check:
            assert x.size(2) == self.height and x.size(3) == self.width

        if not hparams.TRAIN.OFFSET.ALWAYS_TRAIN_BLOCK and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        out_pre = self.pre_offset(x)

        if self.dpool:
            out_pre = self.dpool(out_pre)

        out_dis = self.displace(out_pre, transformer_source=x)

        if self.atten_displace is not None:
            out_atten = self.atten_displace(x)
        else:
            out_atten = None
        out_post = self.post_offset(out_atten * out_dis if out_atten is not None else out_dis)
        if self.downsample is not None:
            x = self.downsample(x)
        out_skip = x + (out_post * self.atten_post(x) if self.atten_post is not None else out_post)

        out_final = self.relu(self.bn(out_skip))

        return out_final

class ConvBlockWithAtten(nn.Module):
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1):
        super(ConvBlockWithAtten, self).__init__()
        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        # self.displace_planes = displace_planes
        self.displace_planes = inplanes // 2
        self.outplanes = outplanes
        self.stride = stride
        self.pre_poffset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_poffset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        self.atten_pdisplace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=True, stride=stride)
        self.pdisplace = nn.Conv2d(self.displace_planes, self.displace_planes, (3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        dis = self.post_poffset(self.pdisplace(self.pre_poffset(x)) * self.atten_pdisplace(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(self.bn(x + dis))
