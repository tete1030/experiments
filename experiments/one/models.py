import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.displacechan import DisplaceChannel, OffsetTransformer, PositionalGaussianDisplaceModule
from lib.models.spacenorm import SpaceNormalization
from utils.globals import config, hparams, globalvars
from utils.log import log_i

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
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN),
                nn.Softplus(),
                SpaceNormalization())
        else:
            self.atten = nn.Sequential(
                nn.Conv2d(self.total_inplanes, outplanes, 1, stride=stride),
                nn.BatchNorm2d(outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN),
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

class SequentialForOffsetBlockTransformer(nn.Sequential):
    def forward(self, input_, extra, collect_outputs=False):
        if collect_outputs:
            outputs = []
        out = input_
        for module in self._modules.values():
            if isinstance(module, OffsetBlock):
                out = module(out, transformer_source=extra)
            else:
                out = module(out)

            if collect_outputs:
                outputs.append(out)
        if collect_outputs:
            return outputs
        else:
            return out

class OffsetBlock(nn.Module):
    def __init__(
            self, height, width, inplanes, outplanes, displace_planes,
            stride=1,
            use_transformer=False,
            use_atten=False,
            atten_source="input",
            atten_space_norm=False,
            use_post_atten=False,
            post_atten_source="input",
            post_atten_space_norm=False,
            post_groups=1,
            num_transform=None,
            disable_arc=False):
        super(OffsetBlock, self).__init__()
        self.height = height
        self.width = width
        self.out_height = (height + stride - 1) // stride
        self.out_width = (width + stride - 1) // stride
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.stride = stride
        if hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE > 0:
            init_num_x = 2 * self.out_width // hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE
            init_num_y = 2 * self.out_height // hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE
            assert init_num_x > 0 and init_num_y > 0, "Insufficient number of init offsets"
            init_num_off = init_num_x * init_num_y
            channels_per_off = int(np.round(displace_planes / init_num_off).item())
            displace_planes_new = channels_per_off * init_num_off
            log_i("Displace plane number rounded from {} to {}".format(displace_planes, displace_planes_new))
            displace_planes = displace_planes_new
        self.displace_planes = displace_planes
        self.use_transformer = use_transformer
        num_offset = self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN
        if use_transformer:
            offset_transformer = OffsetTransformer(
                hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT else self.inplanes,
                num_offset,
                bottleneck=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.BOTTLENECK,
                num_regress=num_transform,
                scale_grow_step=1 / hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER if hparams.TRAIN.OFFSET.TRANSFORMER_GROW_ITER > 0 else None,
                absolute_regressor=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ABSOLUTE_REGRESSOR)
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
            learnable_offset=hparams.MODEL.DETAIL.LEARNABLE_OFFSET,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            transformer=offset_transformer,
            half_reversed_offset=hparams.MODEL.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=globalvars.displace_mods[-1] if hparams.MODEL.LEARNABLE_OFFSET.REUSE_OFFSET and len(globalvars.displace_mods) > 0 else None,
            arc_gaussian=arc_displacer)

        assert not (hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE > 0 and hparams.MODEL.LEARNABLE_OFFSET.INIT_RANDOM_SCALE > 0)
        if hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE > 0:
            width_span = hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE * (init_num_x - 1)
            height_span = hparams.MODEL.LEARNABLE_OFFSET.INIT_STRIDE * (init_num_y - 1)
            for iy, off_y in enumerate(torch.linspace(-height_span / 2, height_span / 2, steps=init_num_y)):
                for ix, off_x in enumerate(torch.linspace(-width_span / 2, width_span / 2, steps=init_num_x)):
                    ichan = (iy * init_num_x + ix) * channels_per_off
                    self.displace.offset.data[ichan:ichan+channels_per_off, 0] = off_x / self.displace.offset_scale
                    self.displace.offset.data[ichan:ichan+channels_per_off, 1] = off_y / self.displace.offset_scale
        if hparams.MODEL.LEARNABLE_OFFSET.INIT_RANDOM_SCALE > 0:
            self.displace.offset.data.normal_(mean=0, std=hparams.MODEL.LEARNABLE_OFFSET.INIT_RANDOM_SCALE)

        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1, groups=post_groups)

        self._atten_source = atten_source
        if use_atten:
            if atten_source == "input":
                atten_inplanes = self.inplanes
            elif atten_source == "transformer":
                assert use_transformer and hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT
                atten_inplanes = hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE
            else:
                raise ValueError("Unknown atten_source = '{}'".format(atten_source))
            self.atten_displace = Attention(
                atten_inplanes,
                self.displace_planes,
                input_shape=(self.height, self.width),
                bias_planes=0,
                bias_factor=0,
                space_norm=atten_space_norm,
                stride=stride)
        else:
            self.atten_displace = None

        self._post_atten_source = post_atten_source
        if use_post_atten:
            if post_atten_source == "input":
                post_atten_inplanes = self.inplanes
            elif post_atten_source == "transformer":
                assert use_transformer and hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT
                post_atten_inplanes = hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE
            else:
                raise ValueError("Unknown post_atten_source = '{}'".format(post_atten_source))
            self.atten_post = Attention(
                post_atten_inplanes,
                self.inplanes,
                input_shape=(self.out_height, self.out_width),
                bias_planes=0,
                bias_factor=0,
                space_norm=post_atten_space_norm,
                stride=stride)
        else:
            self.atten_post = None
        self.bn = nn.BatchNorm2d(self.outplanes, momentum=hparams.TRAIN.OFFSET.MOMENTUM_BN)
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

    def forward(self, x, transformer_source=None):
        if self.use_transformer:
            if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
                assert transformer_source is not None
                actual_transformer_source = transformer_source
            else:
                assert transformer_source is None
                actual_transformer_source = x
        else:
            assert transformer_source is None
            actual_transformer_source = None

        if not hparams.TRAIN.OFFSET.ALWAYS_TRAIN_BLOCK and globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        shortcut = x
        out_pre = self.pre_offset(x)

        if self.dpool:
            out_pre = self.dpool(out_pre)

        out_dis = self.displace(out_pre, transformer_source=actual_transformer_source)

        if self.atten_displace is not None:
            if self._atten_source == "input":
                atten_source = x
            elif self._atten_source == "transformer":
                assert transformer_source is not None
                atten_source = transformer_source
                if atten_source.size()[-2:] != out_dis.size()[-2:]:
                    height_new = out_dis.size(-2)
                    width_new = out_dis.size(-1)
                    height_ori = atten_source.size(-2)
                    width_ori = atten_source.size(-1)
                    assert height_new / height_ori == width_new / width_ori and height_new > height_ori
                    atten_source = F.interpolate(atten_source, size=(height_new, width_new), mode="bilinear", align_corners=True)
            out_atten = self.atten_displace(atten_source)
            out_dis = out_dis * out_atten
        else:
            out_atten = None

        out_post = self.post_offset(out_dis)

        if self.atten_post is not None:
            if self._post_atten_source == "input":
                post_atten_source = x
            elif self._post_atten_source == "transformer":
                assert transformer_source is not None
                post_atten_source = transformer_source
                if post_atten_source.size()[-2:] != out_post.size()[-2:]:
                    height_new = out_post.size(-2)
                    width_new = out_post.size(-1)
                    height_ori = post_atten_source.size(-2)
                    width_ori = post_atten_source.size(-1)
                    assert height_new / height_ori == width_new / width_ori and height_new > height_ori
                    post_atten_source = F.interpolate(post_atten_source, size=(height_new, width_new), mode="bilinear", align_corners=True)
            out_post = out_post * self.atten_post(post_atten_source)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_skip = shortcut + out_post

        out_final = self.relu(self.bn(out_skip))

        return out_final

class MyPose(nn.Module):
    def __init__(self, num_class):
        super(MyPose, self).__init__()
        if hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE and hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
            self.transformer = TransformFeature()
        else:
            self.transformer = None
        self.estimator = SimpleEstimator(num_class)

    def forward(self, x):
        if self.transformer is not None:
            transform_features = self.transformer(x)
        else:
            transform_features = None
        prediction = self.estimator(x, transform_features)
        return prediction

class TransformFeature(nn.Module):
    def __init__(self):
        super(TransformFeature, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        cur_num_channel = 64
        num_out_channel = hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_FEATURE
        offblks = []
        shape_factor = 4
        for i in range(hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_BLK):
            offblks.append(
                OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // shape_factor,
                    hparams.MODEL.INP_SHAPE[0] // shape_factor,
                    cur_num_channel,
                    num_out_channel,
                    hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.NUM_OFFSET[i],
                    stride=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.STRIDE[i],
                    use_atten=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ATTEN.ENABLE,
                    atten_source=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ATTEN.SOURCE,
                    atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ATTEN.SPACE_NORM,
                    use_post_atten=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.POST_ATTEN.ENABLE,
                    post_atten_source=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.POST_ATTEN.SOURCE,
                    post_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.POST_ATTEN.SPACE_NORM,
                    use_transformer=False,
                    disable_arc=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.DISABLE_ARC))
            shape_factor *= 2
            cur_num_channel = num_out_channel
        self.offblk = nn.Sequential(*offblks)

    def forward(self, x):
        return self.offblk(self.pre(x))

class RegressPredictor(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.height = height
        self.width = width
        x = torch.arange(width, dtype=torch.float) + 0.5
        y = torch.arange(height, dtype=torch.float) + 0.5
        self.register_buffer("weight_x", x.view(1, -1).repeat(height, 1))
        self.register_buffer("weight_y", y.view(-1, 1).repeat(1, width))
    
    def forward(self, inp):
        assert inp.size()[-2:] == (self.height, self.width)
        inp_exp_norm = inp.exp()
        inp_exp_norm = inp_exp_norm / inp_exp_norm.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        x = (inp_exp_norm * self.weight_x.expand_as(inp_exp_norm)).sum(dim=-1).sum(dim=-1)
        y = (inp_exp_norm * self.weight_y.expand_as(inp_exp_norm)).sum(dim=-1).sum(dim=-1)
        return torch.stack([x, y], dim=-1)

class SimpleEstimator(nn.Module):
    def __init__(self, num_class):
        super(SimpleEstimator, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        cur_num_channel = 64
        num_out_channel = hparams.MODEL.LEARNABLE_OFFSET.NUM_OUT_CHANNEL
        offblks = []
        predictors = []
        for i in range(hparams.MODEL.LEARNABLE_OFFSET.NUM_BLK):
            offblks.append(
                OffsetBlock(
                    hparams.MODEL.INP_SHAPE[1] // 4,
                    hparams.MODEL.INP_SHAPE[0] // 4,
                    cur_num_channel,
                    num_out_channel,
                    hparams.MODEL.LEARNABLE_OFFSET.NUM_OFFSET[i],
                    num_transform=hparams.MODEL.LEARNABLE_OFFSET.NUM_TRANSFORM[i] if hparams.MODEL.LEARNABLE_OFFSET.NUM_TRANSFORM else None,
                    use_atten=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.ENABLE,
                    atten_source=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SOURCE,
                    atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.ATTEN.SPACE_NORM,
                    use_post_atten=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.ENABLE,
                    post_atten_source=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.SOURCE,
                    post_atten_space_norm=hparams.MODEL.LEARNABLE_OFFSET.POST_ATTEN.SPACE_NORM,
                    post_groups=hparams.MODEL.LEARNABLE_OFFSET.POST_GROUPS[i],
                    use_transformer=hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE))
            cur_num_channel = num_out_channel
            if hparams.MODEL.MULTI_PREDICT:
                predictors.append(self._make_predictor(cur_num_channel, num_class))
        self.offblk = SequentialForOffsetBlockTransformer(*offblks)
        
        if hparams.MODEL.MULTI_PREDICT:
            self.predictor = nn.ModuleList(predictors)
        else:
            self.predictor = self._make_predictor(num_out_channel, num_class)

    def _make_predictor(self, planes, num_class):
        layers = []
        if hparams.MODEL.FC:
            layers.append(nn.Conv2d(planes, planes,
                kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(planes, num_class,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x, transform_features):
        if not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.ENABLE or not hparams.MODEL.LEARNABLE_OFFSET.TRANSFORMER.INDEPENDENT:
            assert transform_features is None
        else:
            assert transform_features is not None

        off = self.offblk(self.pre(x), transform_features, collect_outputs=hparams.MODEL.MULTI_PREDICT)

        if hparams.MODEL.MULTI_PREDICT:
            return list(map(lambda px: px[0](px[1]), zip(self.predictor, off)))
        else:
            return self.predictor(off)
