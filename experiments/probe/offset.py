import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.globals import config, hparams, globalvars
from utils.log import log_i, log_w, log_progress
from lib.models.spacenorm import SpaceNormalization
from lib.models.displacechan import DisplaceChannel, PositionalGaussianDisplaceModule, OffsetTransformer, TransformCoordinate
from .featstab import save_feat_stab

EPS = np.finfo(np.float32).eps.item()

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
    def __init__(self, height, width, inplanes, outplanes, displace_planes,
            use_atten, use_atten_space_norm,
            use_post_atten, use_post_atten_space_norm,
            use_transformer, use_arc,
            dpool_size, always_train_block, use_fusion=False, stride=1,
            use_stabilizer=False):
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

        if use_stabilizer:
            self.stabilizer = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, 1),
                nn.BatchNorm2d(inplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
                nn.ReLU(inplace=True))
            globalvars.exp.feat_stab_stabilizer_parameters += self.stabilizer.parameters()
        else:
            self.stabilizer = None

        num_offset = self.displace_planes
        self.transformer_regressor = None
        if use_transformer:
            offset_transformer = OffsetTransformer(
                num_offset,
                init_effect_scale=0. if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None)
            if not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
                self.transformer_regressor = LocalTransformerRegressor(self.height, self.width, self.inplanes)
                globalvars.exp.feat_stab_displace_parameters += self.transformer_regressor.parameters()
            else:
                self.transformer_regressor = IndpendentTransformerRegressorDelegate(self.inplanes)
                raise NotImplementedError("feat_stab for independent transformer not implemented")
        else:
            offset_transformer = None

        if use_arc:
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
                learnable_sigma=hparams.MODEL.LEARNABLE_OFFSET.ARC.OPTIMIZE_SIGMA,
                transform_sigma=False,
                soft_maxpool=hparams.MODEL.LEARNABLE_OFFSET.ARC.SOFT_MAXPOOL,
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
        if use_fusion:
            self.post_offset = nn.Sequential(
                nn.Conv2d(self.displace_planes, self.displace_planes // 4, 1, groups=self.displace_planes // 4),
                nn.Conv2d(self.displace_planes // 4, self.outplanes, 1))
        else:
            self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if use_atten:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=use_atten_space_norm, stride=stride)
        else:
            self.atten_displace = None
        if use_post_atten:
            self.atten_post = Attention(self.inplanes, self.outplanes, input_shape=(self.out_height, self.out_width), bias_planes=0, bias_factor=0, space_norm=use_post_atten_space_norm, stride=stride)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

        if dpool_size:
            self.dpool = DynamicPooling(self.displace_planes, dpool_size)
        else:
            self.dpool = None

    def forward(self, x):
        if config.check:
            assert x.size(2) == self.height and x.size(3) == self.width

        if not self.always_train_block and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        use_feat_stab = bool(self.stabilizer is not None)
        grad_enable = torch.is_grad_enabled() or (use_feat_stab and bool(globalvars.feat_stab_running))

        if self.stabilizer is not None:
            feat_stab_training = use_feat_stab and bool(self.training and globalvars.feat_stab_running)
            if feat_stab_training:
                self.stabilizer.train(False)
            with torch.autograd.set_grad_enabled(grad_enable):
                stab = self.stabilizer(x)
            if feat_stab_training:
                self.stabilizer.train(True)
            
            if grad_enable and use_feat_stab:
                save_feat_stab("stabilizer", stab)
        else:
            stab = x

        out_pre = self.pre_offset(stab)

        if self.dpool:
            out_pre = self.dpool(out_pre)

        with torch.autograd.set_grad_enabled(grad_enable):
            if self.transformer_regressor:
                kcos, ksin = self.transformer_regressor(stab)
                out_dis = self.displace(out_pre, transformer_kcos=kcos, transformer_ksin=ksin)
            else:
                out_dis = self.displace(out_pre)

            if self.atten_displace is not None:
                out_atten = self.atten_displace(stab)
                out_dis = out_atten * out_dis

            if grad_enable and use_feat_stab:
                assert self.transformer_regressor is not None
                save_feat_stab("displace", out_dis)

        out_post = self.post_offset(out_dis)
        if self.downsample is not None:
            x = self.downsample(x)
        out_skip = x + (out_post * self.atten_post(stab) if self.atten_post is not None else out_post)

        out_final = self.relu(self.bn(out_skip))

        return out_final

class ActiveConv(nn.Module):
    def __init__(self, height, width, inplanes, outplanes, offset_transformer, arc_displacer, dpool, bias=True):
        super().__init__()
        self.num_offsets = inplanes * hparams.MODEL.ACTIVE_BLOCK.OFFSET_PER_CHANNEL
        self.displace = DisplaceChannel(
            height, width,
            self.num_offsets, self.num_offsets,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=True,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transformer=offset_transformer,
            arc_gaussian=arc_displacer)
        globalvars.displace_mods.append(self.displace)
        self.dpool = dpool
        self.conv = nn.Conv2d(self.num_offsets, outplanes, 1, bias=bias)

    def forward(self, inp, transformer_kcos=None, transformer_ksin=None):
        out_pre = inp\
            .view(inp.size(0), inp.size(1), 1, inp.size(2), inp.size(3))\
            .repeat(1, 1, hparams.MODEL.ACTIVE_BLOCK.OFFSET_PER_CHANNEL, 1, 1)\
            .view(inp.size(0), -1, inp.size(2), inp.size(3))
        if self.dpool:
            out_pre = self.dpool(out_pre)

        out_dis = self.displace(out_pre, transformer_kcos=transformer_kcos, transformer_ksin=transformer_ksin)
        out = self.conv(out_dis)

        return out

class ActiveBlock(nn.Module):
    def __init__(self, height, width, inplanes, outplanes, conv_planes,
            use_atten, use_atten_space_norm,
            use_transformer, use_arc,
            dpool_size, always_train_block, stride=1,
            use_stabilizer=False):
        super(ActiveBlock, self).__init__()

        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.conv_planes = conv_planes
        self.num_offset = conv_planes * hparams.MODEL.ACTIVE_BLOCK.OFFSET_PER_CHANNEL
        self.outplanes = outplanes
        self.stride = stride
        self.always_train_block = always_train_block
        if config.check:
            log_i("inplanes={}, outplanes={}, in_height={}, in_width={}, out_height={}, out_width={}, conv_planes={}, num_offsets={}".format(
                self.inplanes,
                self.outplanes,
                self.height,
                self.width,
                self.out_height,
                self.out_width,
                self.conv_planes,
                self.num_offset
            ))

        if use_stabilizer:
            self.stabilizer = nn.Sequential(
                nn.Conv2d(inplanes, inplanes, 1),
                nn.BatchNorm2d(inplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM),
                nn.ReLU(inplace=True))
            globalvars.exp.feat_stab_stabilizer_parameters += self.stabilizer.parameters()
        else:
            self.stabilizer = None

        self.transformer_regressor = None
        if use_transformer:
            offset_transformer = OffsetTransformer(
                self.num_offset,
                init_effect_scale=0. if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None)
            if not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
                self.transformer_regressor = LocalTransformerRegressor(self.height, self.width, self.inplanes)
                globalvars.exp.feat_stab_displace_parameters += self.transformer_regressor.parameters()
            else:
                self.transformer_regressor = IndpendentTransformerRegressorDelegate(self.inplanes)
                raise NotImplementedError("feat_stab for independent transformer not implemented")
        else:
            offset_transformer = None

        if use_arc:
            arc_displacer = PositionalGaussianDisplaceModule(
                self.num_offset,
                hparams.MODEL.LEARNABLE_OFFSET.ARC.NUM_SAMPLE,
                float(hparams.MODEL.LEARNABLE_OFFSET.ARC.ANGLE_STD) / 180 * np.pi,
                hparams.MODEL.LEARNABLE_OFFSET.ARC.SCALE_STD,
                min_angle_std=float(hparams.MODEL.LEARNABLE_OFFSET.ARC.MIN_ANGLE_STD) / 180 * np.pi,
                min_scale_std=hparams.MODEL.LEARNABLE_OFFSET.ARC.MIN_SCALE_STD,
                max_scale_std=5.1,
                sampler=hparams.MODEL.LEARNABLE_OFFSET.ARC.SAMPLER,
                weight_dist=hparams.MODEL.LEARNABLE_OFFSET.ARC.WEIGHT_DIST,
                learnable_sigma=hparams.MODEL.LEARNABLE_OFFSET.ARC.OPTIMIZE_SIGMA,
                transform_sigma=False,
                soft_maxpool=hparams.MODEL.LEARNABLE_OFFSET.ARC.SOFT_MAXPOOL,
                simple=hparams.MODEL.LEARNABLE_OFFSET.ARC.SIMPLE)
            globalvars.arc_displacers.append(arc_displacer)
        else:
            arc_displacer = None

        # NOTE: Here we use dpool on offset
        if dpool_size:
            dpool = DynamicPooling(self.num_offset, dpool_size)
        else:
            dpool = None
        
        self.aconv = ActiveConv(
            self.out_height, self.out_width,
            self.conv_planes, self.conv_planes,
            offset_transformer=offset_transformer,
            arc_displacer=arc_displacer,
            dpool=dpool)
        self.pre_offset = nn.Conv2d(self.inplanes, self.conv_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.conv_planes, self.outplanes, 1)
        # NOTE: Here we only use attention on conv_planes
        if use_atten:
            self.atten = Attention(self.inplanes, self.conv_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=use_atten_space_norm, stride=stride)
        else:
            self.atten = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        if stride > 1 or inplanes != outplanes:
            self.downsample = nn.Conv2d(self.inplanes, self.outplanes,
                          kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        if config.check:
            assert x.size(2) == self.height and x.size(3) == self.width

        if not self.always_train_block and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        use_feat_stab = bool(self.stabilizer is not None)
        grad_enable = torch.is_grad_enabled() or (use_feat_stab and bool(globalvars.feat_stab_running))

        if self.stabilizer is not None:
            feat_stab_training = use_feat_stab and bool(self.training and globalvars.feat_stab_running)
            if feat_stab_training:
                self.stabilizer.train(False)
            with torch.autograd.set_grad_enabled(grad_enable):
                stab = self.stabilizer(x)
            if feat_stab_training:
                self.stabilizer.train(True)

            if grad_enable and use_feat_stab:
                save_feat_stab("stabilizer", stab)
        else:
            stab = x

        out_pre = self.pre_offset(stab)

        with torch.autograd.set_grad_enabled(grad_enable):
            if self.transformer_regressor:
                kcos, ksin = self.transformer_regressor(stab)
                out_dis = self.aconv(out_pre, transformer_kcos=kcos, transformer_ksin=ksin)
            else:
                out_dis = self.aconv(out_pre)

            if self.atten is not None:
                out_atten = self.atten(stab)
                out_dis = out_dis * out_atten

            if grad_enable and use_feat_stab:
                assert self.transformer_regressor is not None
                save_feat_stab("displace", out_dis)

        out_post = self.post_offset(out_dis)
        if self.downsample is not None:
            x = self.downsample(x)
        out_skip = x + out_post

        out_final = self.relu(self.bn(out_skip))

        return out_final

class Probe(nn.Module):
    def __init__(self, height, width, inplanes, num_offsets, num_probes, probe_min, probe_max, probe_type, dpool_size):
        super().__init__()
        assert probe_type in ["scale", "angle"]
        total_offsets = num_offsets * num_probes
        self.single_source = False
        self.use_softmax = True
        if self.single_source:
            if dpool_size:
                self.dpool = DynamicPooling(inplanes, dpool_size)
            else:
                self.dpool = None
            self.conv = nn.Conv2d(inplanes, 1, 1)
        else:
            self.conv = nn.Conv2d(inplanes, num_offsets, 1)
            if dpool_size:
                self.dpool = DynamicPooling(num_offsets, dpool_size)
            else:
                self.dpool = None

        self.displace = DisplaceChannel(
            height, width,
            total_offsets, total_offsets,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=True,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            runtime_offset=True)
        globalvars.displace_mods.append(self.displace)
        self.offsets = nn.Parameter(torch.randn(num_offsets, 2) * 0.1)
        self.summarizer = nn.Conv2d(num_offsets, 1, 1)
        self.register_buffer("probe_vals", torch.linspace(probe_min, probe_max, num_probes))
        self.num_offsets = num_offsets
        self.num_probes = num_probes
        self.probe_min = probe_min
        self.probe_max = probe_max
        self.probe_type = probe_type

    def forward(self, x):
        if self.probe_type == "scale":
            offsets = (self.offsets.view(1, -1, 2) * self.probe_vals.view(-1, 1, 1)).view(-1, 2)
        elif self.probe_type == "angle":
            kcos = self.probe_vals.cos().view(-1, 1)
            ksin = self.probe_vals.sin().view(-1, 1)
            offsets = torch.stack(
                TransformCoordinate.apply(self.offsets[None, :, 0], self.offsets[None, :, 1], ksin, kcos),
                dim=2).view(-1, 2)

        if self.single_source:
            if self.dpool:
                x = self.dpool(x)
            x = self.conv(x)
            dis = self.displace(x.repeat(1, offsets.size(0), 1, 1), offset_runtime_rel=offsets)
        else:
            x = self.conv(x)
            if self.dpool:
                x = self.dpool(x)
            dis = self.displace(x.repeat(1, self.probe_vals.size(0), 1, 1), offset_runtime_rel=offsets)

        dis = dis.view(x.size(0) * self.num_probes, self.num_offsets, x.size(2), x.size(3))
        out = self.summarizer(dis).view(x.size(0), self.num_probes, x.size(2), x.size(3))

        if self.use_softmax:
            score = torch.nn.functional.softmax(out, dim=1)
        else:
            score = torch.nn.functional.softplus(out) + EPS
            score = score / score.sum(dim=1, keepdim=True)

        probe_val = (score * self.probe_vals.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)

        return probe_val

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
                    use_atten=hparams.MODEL.IND_TRANSFORMER.ATTEN.ENABLE,
                    use_atten_space_norm=hparams.MODEL.IND_TRANSFORMER.ATTEN.SPACE_NORM,
                    use_post_atten=hparams.MODEL.IND_TRANSFORMER.POST_ATTEN.ENABLE,
                    use_post_atten_space_norm=hparams.MODEL.IND_TRANSFORMER.POST_ATTEN.SPACE_NORM,
                    use_transformer=False,
                    use_arc=hparams.MODEL.IND_TRANSFORMER.ENABLE_ARC,
                    dpool_size=hparams.MODEL.IND_TRANSFORMER.DPOOL_SIZE,
                    always_train_block=True,
                    stride=hparams.MODEL.IND_TRANSFORMER.STRIDE[i]))
            shape_factor *= hparams.MODEL.IND_TRANSFORMER.STRIDE[i]
            cur_num_channel = num_out_channel
        self.offblk = nn.Sequential(*offblks)
        self.shape_factor = shape_factor

    def forward(self, x):
        return self.offblk(self.pre(x))

class TransformerHead(nn.Module):
    def __init__(self, height, width, num_channels, detach=True, sep_scale=False):
        super().__init__()
        self.sep_scale = sep_scale
        self.scale_regressor = Probe(
            height, width, num_channels,
            hparams.MODEL.PROBE.NUM_OFFSETS,
            hparams.MODEL.PROBE.NUM_SCALES,
            hparams.MODEL.PROBE.SCALE_MIN,
            hparams.MODEL.PROBE.SCALE_MAX,
            "scale",
            hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE)
        self.angle_regressor = Probe(
            height, width, num_channels,
            hparams.MODEL.PROBE.NUM_OFFSETS,
            hparams.MODEL.PROBE.NUM_ANGLES,
            hparams.MODEL.PROBE.ANGLE_MIN / 180 * np.pi,
            hparams.MODEL.PROBE.ANGLE_MAX / 180 * np.pi,
            "angle",
            hparams.MODEL.LEARNABLE_OFFSET.DPOOL_SIZE)
        self.detach = detach

    def forward(self, x):
        if self.detach:
            x = x.detach()
        scale = self.scale_regressor(x)
        angle = self.angle_regressor(x)

        angle_cos = angle.cos().clamp(-1+EPS, 1-EPS)
        angle_sin = angle.sin().clamp(-1+EPS, 1-EPS)

        if not self.sep_scale:
            return angle_cos * scale, angle_sin * scale
        else:
            return angle_cos, angle_sin, scale

class IndpendentTransformerRegressor(nn.Module):
    def __init__(self, num_feature, sep_scale=False):
        super().__init__()
        self.transform_feature = TransformerFeature(num_feature)
        self.transform_head = TransformerHead(
            hparams.MODEL.INP_SHAPE[1] // self.transform_feature.shape_factor,
            hparams.MODEL.INP_SHAPE[0] // self.transform_feature.shape_factor,
            num_feature, detach=False, sep_scale=sep_scale)

    def forward(self, x):
        x = self.transform_feature(x)
        return self.transform_head(x)

class IndpendentTransformerRegressorDelegate(nn.Module):
    def __init__(self, num_feature):
        super().__init__()

    def forward(self, x):
        return globalvars.transformer_output[x.device]

class LocalTransformerRegressor(nn.Module):
    def __init__(self, height, width, num_feature):
        super().__init__()
        self.transform_head = TransformerHead(
            height,
            width,
            num_feature)

    def forward(self, x):
        return self.transform_head(x)
