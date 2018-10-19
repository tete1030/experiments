import math
import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars
from utils.log import log_i
import torch.nn.functional as F
import numpy as np
from .displace import Displace, DisplaceCUDA, CustomizedGradDepthwiseConv2d
from pose.models.common import StrictNaNReLU
from utils.lambdalayer import Lambda

class Weighted(nn.Module):
    def __init__(self, num_channels, init=0.):
        super(Weighted, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_channels, dtype=torch.float))
        if isinstance(init, (float, int)):
            self.weight.data.fill_(float(init))
        elif isinstance(init, torch.FloatTensor):
            self.weight.data.copy_(init)
        else:
            raise ValueError("Wrong initialization")

    def forward(self, x):
        assert x.dim() == 2
        return x * self.weight[None]

class OffsetRegressor(nn.Module):
    def __init__(self, inplanes, regressor_channels):
        super(OffsetRegressor, self).__init__()
        self.inplanes = inplanes
        self.atten_inplanes = inplanes // 4
        self.regressor_channels = regressor_channels
        self.pre = nn.Sequential(
            nn.Conv2d(self.inplanes, self.atten_inplanes, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.atten_inplanes),
            StrictNaNReLU(inplace=True),
            nn.Conv2d(self.atten_inplanes, self.atten_inplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.atten_inplanes),
            StrictNaNReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Linear(self.atten_inplanes, regressor_channels, bias=False),
            nn.BatchNorm1d(regressor_channels, affine=False),
            Weighted(regressor_channels, init=0.),
            Lambda(lambda x: x.view(x.size(0), -1, 2)))

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.parameters():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n / 100))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.weight.size(0) / 100))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, atten=None):
        x = self.pre(x)
        if atten is not None:
            x = (x * atten).sum(-1).sum(-1)
        else:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)
        return self.regressor(x)

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, init_stride, chan_per_init_pos,
                 learnable_offset=False, LO_kernel_size=3, LO_sigma=0.5,
                 disable_displace=False, random_offset_init=None, use_origin=False, actual_stride=None,
                 displace_size=None, LO_balance_grad=True, free_offset_per_init_pos=1,
                 dconv_for_LO_stride=1, regress_offset=False,
                 LO_grad_inside_only=False, LO_half_reversed_offset=False,
                 LO_interpolate_kernel_type="gaussian",
                 previous_dischan=None):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.scale = float(max(width, height))
        self.init_stride = init_stride
        self.chan_per_init_pos = chan_per_init_pos
        self.learnable_offset = learnable_offset
        self.disable_displace = disable_displace
        self.random_offset_init = random_offset_init
        self.use_origin = use_origin
        self.actual_stride = actual_stride
        self.displace_size = displace_size
        self.free_offset_per_init_pos = free_offset_per_init_pos
        self.dconv_for_LO_stride = dconv_for_LO_stride
        self.LO_grad_inside_only = LO_grad_inside_only
        self.LO_half_reversed_offset = LO_half_reversed_offset
        self.num_init_y, self.num_init_x, self.num_init_pos = self.get_num_offset(height, width, displace_size, init_stride, use_origin)
        self.inplanes = self.num_init_pos * self.chan_per_init_pos
        self.regress_offset = regress_offset

        if not disable_displace:
            self.previous_dischan = previous_dischan
            previous_offset_channels = 0
            if previous_dischan is not None:
                assert isinstance(previous_dischan, DisplaceChannel)
                assert self.random_offset_init is not None
                previous_offset_channels = self.previous_dischan.get_all_offsets_num()

            assert self.num_init_pos * self.free_offset_per_init_pos >= previous_offset_channels
            self.offset = nn.parameter.Parameter(torch.Tensor(self.num_init_pos * self.free_offset_per_init_pos - previous_offset_channels, 2), requires_grad=False)
            self.init_offset()
            if learnable_offset:
                assert isinstance(LO_kernel_size, int)
                assert LO_kernel_size % 2 == 1

                self.LO_interpolate_kernel_type = LO_interpolate_kernel_type
                self.LO_kernel_size_init = LO_kernel_size
                self.LO_sigma_init = LO_sigma
                self.set_learnable_offset_para(LO_kernel_size, LO_sigma)
                self.LO_balance_grad = LO_balance_grad
                self.switch_LO_state(True)
                self.offset.requires_grad = True
                if LO_balance_grad and self.offset.size(0) > 0:
                    self.offset.register_hook(self.balance_offset_grad)

                if regress_offset:
                    regressor_channels = self.num_init_pos * self.free_offset_per_init_pos * 2
                    self.offset_regressor = OffsetRegressor(self.inplanes, regressor_channels)
                else:
                    self.offset_regressor = None
            else:
                self.switch_LO_state(False)

    def set_learnable_offset_para(self, kernel_size, sigma):
        assert not hasattr(self, "LO_kernel_size") or self.LO_interpolate_kernel_type == "gaussian"
        # TODO: Log
        log_i("learnable offset para set to kernel_size={}, sigma={}".format(kernel_size, sigma))
        assert kernel_size % 2 == 1
        self.LO_kernel_size = kernel_size
        self.LO_sigma = sigma

        x = torch.arange(kernel_size, dtype=torch.float).view(1, -1, 1).expand(kernel_size, -1, -1) - float(kernel_size // 2)
        y = torch.arange(kernel_size, dtype=torch.float).view(-1, 1, 1).expand(-1, kernel_size, -1) - float(kernel_size // 2)
        field = torch.cat([x, y], dim=2).expand(1, -1, -1, -1)\
            .repeat(self.num_init_pos * self.free_offset_per_init_pos, 1, 1, 1)
        if self.LO_half_reversed_offset:
            field = field.repeat(2, 1, 1, 1)
        self.field = dict()
        self.field[torch.device("cpu")] = field

    def reset_learnable_offset_para(self):
        self.set_learnable_offset_para(self.LO_kernel_size_init, self.LO_sigma_init)

    def switch_LO_state(self, active):
        if active:
            log_i("Learnable offset is enabled")
        else:
            log_i("Learnable offset is disabled")
        self.LO_active = active

    def balance_offset_grad(self, grad):
        area = (self.width - self.scale * self.offset.data[:, 0].round().abs()) * (self.height - self.scale * self.offset.data[:, 1].round().abs())
        return grad / area.view(-1, 1)

    @staticmethod
    def get_num_offset(height, width, displace_size, init_stride, use_origin):
        if displace_size:
            num_x = displace_size[0]
            num_y = displace_size[1]
        else:
            assert int(height) - init_stride > init_stride
            assert int(width) - init_stride > init_stride
            num_y = (int(height) - init_stride) // init_stride * 2 + 1
            num_x = (int(width) - init_stride) // init_stride * 2 + 1

        num_pos = num_y * num_x
        if not use_origin:
            num_pos -= 1
        return num_y, num_x, num_pos

    def init_offset(self):
        nh, nw = self.num_init_y, self.num_init_x
        if self.random_offset_init is not None:
            self.offset.data.uniform_(-self.random_offset_init, self.random_offset_init)
            return

        assert self.offset.size(0) == self.num_init_pos * self.free_offset_per_init_pos
        count_off = 0
        for ih in range(-(nh // 2), nh // 2 + 1):
            for iw in range(-(nw // 2), nw // 2 + 1):
                if not self.use_origin and ih == 0 and iw == 0:
                    continue
                if not self.actual_stride:
                    self.offset.data[count_off*self.free_offset_per_init_pos:(count_off+1)*self.free_offset_per_init_pos, 0] = iw * self.init_stride / self.scale
                    self.offset.data[count_off*self.free_offset_per_init_pos:(count_off+1)*self.free_offset_per_init_pos, 1] = ih * self.init_stride / self.scale
                else:
                    self.offset.data[count_off*self.free_offset_per_init_pos:(count_off+1)*self.free_offset_per_init_pos, 0] = iw * self.actual_stride / self.scale
                    self.offset.data[count_off*self.free_offset_per_init_pos:(count_off+1)*self.free_offset_per_init_pos, 1] = ih * self.actual_stride / self.scale
                count_off += 1

    def reset_outsider(self):
        if self.offset.size(0) == 0:
            return

        self.offset.data[:, 0].clamp_(min=(-self.width + 0.6) / self.scale,
                                      max=(self.width - 0.6) / self.scale)
        self.offset.data[:, 1].clamp_(min=(-self.height + 0.6) / self.scale,
                                      max=(self.height - 0.6) / self.scale)

    def interpolate(self, x, offset_rel):
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        free_offsets = offset_rel.size(-2)
        bind_chan = num_channels // free_offsets
        field = self.field[x.device]
        suboffset_rel = offset_rel - offset_rel.detach().round()
        if offset_rel.dim() == 3:
            if self.LO_interpolate_kernel_type == "gaussian":
                # Dynamic offset
                kernel = torch.exp(- (field[None] + suboffset_rel[:, :, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
                kernel = kernel / kernel.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
            elif self.LO_interpolate_kernel_type == "bilinear":
                kernel = (1 - (field[None] + suboffset_rel[:, :, None, None, :]).abs()).clamp(min=0).prod(dim=-1)
            # nsamp*npos*bind_chan x kh x kw
            kernel = kernel.view(batch_size*free_offsets, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, bind_chan, 1, 1).view(batch_size*num_channels, 1, self.LO_kernel_size, self.LO_kernel_size)
            # 1 x nsamp*npos*bind_chan x height x width
            out = CustomizedGradDepthwiseConv2d.apply(
                x.view(1, batch_size*num_channels, height, width), kernel, None,
                (1, 1),
                (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride),
                (self.dconv_for_LO_stride, self.dconv_for_LO_stride),
                batch_size*num_channels,
                offset_rel.detach().round().int().view(-1, 2) if self.LO_grad_inside_only else None,
                bind_chan,
                (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride))
            out = out.view(batch_size, num_channels, height, width)
            return out
        elif offset_rel.dim() == 2:
            # Static offset
            if self.LO_interpolate_kernel_type == "gaussian":
                # The offset means which direction and how long we should move the image,
                # while for each kernel at each individual place, -offset is the center where it should 'look at',
                # so the field becomes to force - (-offset.float) -> force + offset.float
                kernel = torch.exp(- (field + suboffset_rel[:, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
                kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
            elif self.LO_interpolate_kernel_type == "bilinear":
                kernel = (1 - (field + suboffset_rel[:, None, None, :]).abs()).clamp(min=0).prod(dim=-1)
            # npos*bind_chan x kh x kw
            kernel = kernel.view(free_offsets, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, bind_chan, 1, 1).view(num_channels, 1, self.LO_kernel_size, self.LO_kernel_size)
            # nsamp x npos*bind_chan x height x width
            out = CustomizedGradDepthwiseConv2d.apply(
                x, kernel, None,
                (1, 1),
                (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride),
                (self.dconv_for_LO_stride, self.dconv_for_LO_stride),
                num_channels,
                offset_rel.detach().round().int() if self.LO_grad_inside_only else None,
                bind_chan,
                (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride))
            return out
        else:
            raise ValueError()

    def get_all_offsets_num(self):
        offsets_num = self.offset.size(0)
        if self.previous_dischan is not None:
            offsets_num += self.previous_dischan.get_all_offsets_num()
        return offsets_num

    def get_all_offsets(self, detach=False, cat=True):
        if self.offset.size(0) == 0:
            all_offsets = []
        else:
            all_offsets = [self.offset.detach() if detach else self.offset]

        if self.previous_dischan is not None:
            all_offsets = self.previous_dischan.get_all_offsets(detach=detach, cat=False) + all_offsets
            if cat:
                all_offsets = torch.cat(all_offsets, dim=0)
        elif cat:
            all_offsets = all_offsets[0]

        return all_offsets

    def forward(self, inp, LO_active=None, offset_plus_rel=None, offset_regressor_atten=None):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        device = inp.device
        free_offsets = self.num_init_pos * self.free_offset_per_init_pos
        assert self.height == height and self.width == width
        assert num_channels % free_offsets == 0, "num of channels cannot be divided by number of offsets"
        assert offset_plus_rel is None or offset_plus_rel.size(1) == free_offsets

        out_LO = None

        if not self.disable_displace:
            offset_rel = self.get_all_offsets()

            if offset_plus_rel is not None:
                offset_rel = offset_rel[None] + offset_plus_rel

            if self.regress_offset:
                if offset_rel.dim() == 2:
                    offset_rel = offset_rel[None]
                if offset_regressor_atten is not None:
                    offset_regressed = self.offset_regressor(inp, atten=offset_regressor_atten)
                else:
                    offset_regressed = self.offset_regressor(inp)
                offset_rel = offset_rel + offset_regressed

            bind_chan = num_channels // free_offsets
            if self.LO_half_reversed_offset:
                assert bind_chan % 2 == 0
                bind_chan = bind_chan // 2
                offset_rel = torch.cat([offset_rel, -offset_rel], dim=-2)

            offset_abs = offset_rel * self.scale
            if offset_abs.dim() == 3:
                out = DisplaceCUDA.apply(inp.view(1, -1, height, width), offset_abs.detach().round().int().view(-1, 2), bind_chan).view(batch_size, -1, height, width)
            elif offset_abs.dim() == 2:
                out = DisplaceCUDA.apply(inp, offset_abs.detach().round().int(), bind_chan)
            else:
                raise ValueError()

            if self.learnable_offset and (LO_active is True or (LO_active is None and self.LO_active is True)):
                if device not in self.field:
                    self.field[device] = self.field[torch.device("cpu")].to(device)

                out_LO = self.interpolate(out, offset_abs)
        else:
            out = inp

        if config.vis and False:
            import matplotlib.pyplot as plt
            import cv2
            fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
            
            for row, axes_row in enumerate(axes):
                img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = inp.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    if col == 0:
                        ax.imshow(img)
                    else:
                        ax.imshow(fts[col-1])
            fig.suptitle("displace inp")

            fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
            for row, axes_row in enumerate(axes):
                img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = out.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    if col == 0:
                        ax.imshow(img)
                    else:
                        ax.imshow(fts[col-1])
            fig.suptitle("displace out")
            plt.show()

        return out, out_LO
