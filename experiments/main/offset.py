import torch
from torch import nn
from utils.globals import config, hparams, globalvars
from utils.log import log_i, log_w, log_progress
from lib.models.spacenorm import SpaceNormalization
from lib.models.displacechan import DisplaceChannel

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
        self.displace = DisplaceChannel(
            self.out_height, self.out_width,
            self.displace_planes, self.displace_planes // hparams.MODEL.LEARNABLE_OFFSET.BIND_CHAN,
            disable_displace=hparams.MODEL.DETAIL.DISABLE_DISPLACE,
            learnable_offset=hparams.MODEL.DETAIL.DISPLACE_LEARNABLE_OFFSET,
            regress_offset=hparams.MODEL.LEARNABLE_OFFSET.REGRESS_OFFSET,
            half_reversed_offset=hparams.MODEL.LEARNABLE_OFFSET.HALF_REVERSED_OFFSET,
            previous_dischan=globalvars.displace_mods[-1] if hparams.MODEL.LEARNABLE_OFFSET.REUSE_OFFSET and len(globalvars.displace_mods) > 0 else None)
        globalvars.displace_mods.append(self.displace)
        self.pre_offset = nn.Conv2d(self.inplanes, self.displace_planes, 1, stride=stride)
        self.post_offset = nn.Conv2d(self.displace_planes, self.outplanes, 1)
        if hparams.MODEL.LEARNABLE_OFFSET.ENABLE_ATTEN:
            self.atten_displace = Attention(self.inplanes, self.displace_planes, input_shape=(self.height, self.width), bias_planes=0, bias_factor=0, space_norm=True, stride=stride)
        else:
            self.atten_displace = None
        if hparams.MODEL.LEARNABLE_OFFSET.ENABLE_MASK:
            self.atten_post = Attention(0, self.outplanes, input_shape=(self.out_height, self.out_width), bias_planes=inplanes // 4, bias_factor=2, space_norm=False)
        else:
            self.atten_post = None
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

        if globalvars.progress["step"] < hparams.TRAIN.OFFSET.TRAIN_MIN_STEP:
            return x

        out_pre = self.pre_offset(x)
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
