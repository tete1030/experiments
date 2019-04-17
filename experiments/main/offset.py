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
    def __init__(self, height, width, inplanes, outplanes, displace_planes, stride=1,
            disable_atten=False, disable_post_atten=False, disable_transformer=False, disable_arc=False,
            disable_dpool=False, always_train_block=False):
        super(OffsetBlock, self).__init__()

        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.displace_planes = displace_planes
        self.outplanes = outplanes
        self.stride = stride
        self.always_train_block = always_train_block
        if config.check:
            log_i("inplanes={}, outplanes={}, in_height={}, in_width={}, out_height={}, out_width={}, displace_planes={}, num_offsets={}".format(
                self.inplanes,
                self.outplanes,
                self.height,
                self.width,
                self.out_height,
                self.out_width,
                self.displace_planes,
                self.displace_planes
            ))

        num_offset = self.displace_planes
        self.transformer_regressor = None
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE and not disable_transformer:
            offset_transformer = OffsetTransformer(
                num_offset,
                init_effect_scale=0. if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None)
            if not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
                self.transformer_regressor = LocalTransformerRegressor(self.inplanes, hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_REGRESS)
            else:
                self.transformer_regressor = IndpendentTransformerRegressorDelegate(self.inplanes, hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_REGRESS)
        else:
            offset_transformer = None

        if hparams.MODEL.LEARNABLE_OFFSET.ARC.ENABLE and not disable_arc:
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
            learnable_offset=True,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transformer=offset_transformer,
            arc_gaussian=arc_displacer)
        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if hparams.MODEL.LEARNABLE_OFFSET.ATTEN.ENABLE and not disable_atten:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SPACE_NORM, stride=stride)
        else:
            self.atten_displace = None
        if hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.ENABLE and not disable_post_atten:
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

        if hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE > 1 and not disable_dpool:
            self.dpool = DynamicPooling(self.displace_planes, hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE)
        else:
            self.dpool = None

    def forward(self, x):
        if config.check:
            assert x.size(2) == self.height and x.size(3) == self.width

        if not self.always_train_block and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        out_pre = self.pre_offset(x)

        if self.dpool:
            out_pre = self.dpool(out_pre)

        if self.transformer_regressor:
            kcos, ksin = self.transformer_regressor(x)
            out_dis = self.displace(out_pre, transformer_kcos=kcos, transformer_ksin=ksin)
        else:
            out_dis = self.displace(out_pre)

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

class TransformerFeature(nn.Module):
    def __init__(self, num_out_channel):
        super(TransformerFeature, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        cur_num_channel = 64
        offblks = []
        shape_factor = 4
        assert len(hparams.MODEL.IND_TRANSFORMER.STRIDE) == len(hparams.MODEL.IND_TRANSFORMER.NUM_OFFSET) == hparams.MODEL.IND_TRANSFORMER.NUM_BLK
        for i in range(hparams.MODEL.IND_TRANSFORMER.NUM_BLK):
            offblks.append(
                OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // shape_factor,
                    hparams.MODEL.INP_SHAPE[0] // shape_factor,
                    cur_num_channel,
                    num_out_channel,
                    hparams.MODEL.IND_TRANSFORMER.NUM_OFFSET[i],
                    stride=hparams.MODEL.IND_TRANSFORMER.STRIDE[i],
                    disable_atten=False,
                    disable_post_atten=True,
                    disable_arc=True,
                    disable_transformer=True,
                    always_train_block=True))
            shape_factor *= hparams.MODEL.IND_TRANSFORMER.STRIDE[i]
            cur_num_channel = num_out_channel
        self.offblk = nn.Sequential(*offblks)

    def forward(self, x):
        return self.offblk(self.pre(x))

class TransformerHead(nn.Module):
    def __init__(self, num_channels, num_regress, sep_scale=False):
        super().__init__()
        self.sep_scale = sep_scale
        self.pre_scale = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )
        self.scale_regressor = nn.Sequential(
            nn.Conv2d(num_channels, num_regress, kernel_size=1, bias=False),
            nn.Softsign()
        )
        self.pre_angle = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())
        self.angle_x_regressor = nn.Sequential(
            nn.Conv2d(num_channels, num_regress, kernel_size=1, bias=False))
        self.angle_y_regressor = nn.Sequential(
            nn.Conv2d(num_channels, num_regress, kernel_size=1, bias=False))

    def forward(self, x):
        EPS = np.finfo(np.float32).eps.item()
        pre_scale = self.pre_scale(x).clamp(max=88.722835).exp()
        pre_scale = pre_scale / (pre_scale.sum(dim=1, keepdim=True).detach() + EPS)
        scale = 1 + self.scale_regressor(pre_scale)

        pre_angle = self.pre_angle(x).clamp(max=88.722835).exp()
        pre_angle = pre_angle / (pre_angle.sum(dim=1, keepdim=True).detach() + EPS)
        angle_kcos = self.angle_x_regressor(pre_angle)
        angle_ksin = self.angle_y_regressor(pre_angle)

        angle_knorm = (torch.stack([angle_kcos, angle_ksin], dim=0).norm(dim=0) + EPS)
        if not self.sep_scale:
            angle_knorm = angle_knorm / (scale + EPS)
        
        angle_kcos = angle_kcos / angle_knorm
        angle_ksin = angle_ksin / angle_knorm

        if not self.sep_scale:
            return angle_kcos, angle_ksin
        else:
            return angle_kcos, angle_ksin, scale

class IndpendentTransformerRegressor(nn.Module):
    def __init__(self, num_feature, num_regress, sep_scale=False):
        super().__init__()
        self.transform_feature = TransformerFeature(num_feature)
        self.transform_head = TransformerHead(num_feature, num_regress, sep_scale=sep_scale)

    def forward(self, x):
        x = self.transform_feature(x)
        return self.transform_head(x)

class IndpendentTransformerRegressorDelegate(nn.Module):
    def __init__(self, num_feature, num_regress):
        super().__init__()

    def forward(self, x):
        return globalvars.independent_transformer_result[x.device]

class LocalTransformerRegressor(nn.Module):
    def __init__(self, num_feature, num_regress):
        super().__init__()
        self.transform_head = TransformerHead(num_feature, num_regress)

    def forward(self, x):
        return self.transform_head(x)
