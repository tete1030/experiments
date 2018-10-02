import math
import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars
from utils.log import log_i
import torch.nn.functional as F
import numpy as np
from .displace import Displace, DisplaceCUDA
from pose.models.common import StrictNaNReLU

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, init_stride, chan_per_pos,
                 learnable_offset=False, LO_kernel_size=3, LO_sigma=0.5,
                 disable_displace=False, random_offset_init=None, use_origin=False, actual_stride=None,
                 displace_size=None, LO_balance_grad=True, free_chan_per_pos=1,
                 dconv_for_LO_stride=1, regress_offset=False):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.init_stride = init_stride
        self.chan_per_pos = chan_per_pos
        self.learnable_offset = learnable_offset
        self.disable_displace = disable_displace
        self.random_offset_init = random_offset_init
        self.use_origin = use_origin
        self.actual_stride = actual_stride
        self.displace_size = displace_size
        self.free_chan_per_pos = free_chan_per_pos
        self.dconv_for_LO_stride = dconv_for_LO_stride
        self.num_y, self.num_x, self.num_pos = self.get_num_offset(height, width, displace_size, init_stride, use_origin)
        self.inplanes = self.num_pos * self.chan_per_pos
        self.regress_offset = regress_offset

        if not disable_displace:
            self.offset = nn.parameter.Parameter(torch.Tensor(self.num_pos * self.free_chan_per_pos, 2), requires_grad=False)
            self.init_offset()
            if learnable_offset:
                assert isinstance(LO_kernel_size, int)
                assert LO_kernel_size % 2 == 1

                self.LO_kernel_size_init = LO_kernel_size
                self.LO_sigma_init = LO_sigma
                self.set_learnable_offset_para(LO_kernel_size, LO_sigma)
                self.LO_balance_grad = LO_balance_grad
                self.switch_LO_state(True)
                self.offset.requires_grad = True
                if LO_balance_grad:
                    self.offset.register_hook(self.balance_offset_grad)

                if regress_offset:
                    self.offset_regressor = nn.Sequential(
                        nn.Conv2d(self.inplanes, self.inplanes // 4, kernel_size=1, stride=1),
                        nn.BatchNorm2d(self.inplanes // 4),
                        StrictNaNReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 4, self.inplanes // 4, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(self.inplanes // 4),
                        StrictNaNReLU(inplace=True),
                        nn.Conv2d(self.inplanes // 4, self.num_pos * self.free_chan_per_pos * 2, kernel_size=1, stride=1, bias=False))

                    for m in self.offset_regressor:
                        if isinstance(m, nn.Conv2d):
                            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                            m.weight.data.normal_(0, math.sqrt(2. / n / 100))
                        elif isinstance(m, nn.GroupNorm):
                            m.weight.data.fill_(1)
                            m.bias.data.zero_()
                    self.offset_regressor[-1].weight.data.zero_()
            else:
                self.switch_LO_state(False)

    def set_learnable_offset_para(self, kernel_size, sigma):
        # TODO: Log
        log_i("learnable offset para set to kernel_size={}, sigma={}".format(kernel_size, sigma))
        assert kernel_size % 2 == 1
        self.LO_kernel_size = kernel_size
        self.LO_sigma = sigma

        x = torch.arange(kernel_size, dtype=torch.float).view(1, -1, 1).expand(kernel_size, -1, -1) - float(kernel_size // 2)
        y = torch.arange(kernel_size, dtype=torch.float).view(-1, 1, 1).expand(-1, kernel_size, -1) - float(kernel_size // 2)
        self.field = dict()
        self.field[torch.device("cpu")] = torch.cat([x, y], dim=2).expand(1, -1, -1, -1).repeat(self.num_pos * self.free_chan_per_pos, 1, 1, 1)

    def reset_learnable_offset_para(self):
        self.set_learnable_offset_para(self.LO_kernel_size_init, self.LO_sigma_init)

    def switch_LO_state(self, active):
        if active:
            log_i("Learnable offset is enabled")
        else:
            log_i("Learnable offset is disabled")
        self.LO_active = active

    def balance_offset_grad(self, grad):
        area = (self.width - self.offset.data[:, 0].round().abs()) * (self.height - self.offset.data[:, 1].round().abs())
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
        nh, nw = self.num_y, self.num_x
        if self.random_offset_init is not None:
            self.offset.data.uniform_(-self.random_offset_init, self.random_offset_init)
            return
        count_off = 0
        for ih in range(-(nh // 2), nh // 2 + 1):
            for iw in range(-(nw // 2), nw // 2 + 1):
                if not self.use_origin and ih == 0 and iw == 0:
                    continue
                if not self.actual_stride:
                    self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 0] = iw * self.init_stride
                    self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 1] = ih * self.init_stride
                else:
                    self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 0] = iw * self.actual_stride
                    self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 1] = ih * self.actual_stride
                count_off += 1

    def reset_outsider(self):
        self.offset.data[:, 0].clamp_(min=-self.width + 0.6,
                                      max=self.width - 0.6)
        self.offset.data[:, 1].clamp_(min=-self.height + 0.6,
                                      max=self.height - 0.6)

    def gaussian_translate(self, x, offset):
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        chan_group_size = num_channels // (self.num_pos * self.free_chan_per_pos)
        field = self.field[x.device]
        if offset.dim() == 3:
            kernel = torch.exp(- (field[None] + (offset - offset.detach().round())[:, :, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
            kernel = kernel / kernel.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
            # nsamp*npos*chan_group_size x kh x kw
            kernel = kernel.view(batch_size*self.num_pos*self.free_chan_per_pos, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, chan_group_size, 1, 1).view(batch_size*num_channels, 1, self.LO_kernel_size, self.LO_kernel_size)
            # 1 x nsamp*npos*chan_group_size x height x width
            out = F.conv2d(x.view(1, batch_size*num_channels, height, width), kernel, None, (1, 1), (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride), (self.dconv_for_LO_stride, self.dconv_for_LO_stride), batch_size*num_channels)
            out = out.view(batch_size, num_channels, height, width)
            return out
        elif offset.dim() == 2:
            # The offset means which direction and how long we should move the image,
            # while for each kernel at each individual place, -offset is the center where it should 'look at',
            # so the field becomes to force - (-offset.float) -> force + offset.float
            kernel = torch.exp(- (field + (self.offset - self.offset.detach().round())[:, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
            kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
            # npos*chan_group_size x kh x kw
            kernel = kernel.view(self.num_pos*self.free_chan_per_pos, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, chan_group_size, 1, 1).view(num_channels, 1, self.LO_kernel_size, self.LO_kernel_size)
            # nsamp x npos*chan_group_size x height x width
            out = F.conv2d(x, kernel, None, (1, 1), (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride), (self.dconv_for_LO_stride, self.dconv_for_LO_stride), num_channels)
            return out
        else:
            raise ValueError()

    def forward(self, inp, LO_active=None, offset_plus=None):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        device = inp.device
        free_channels = self.num_pos * self.free_chan_per_pos
        assert self.height == height and self.width == width
        assert num_channels % free_channels == 0, "num of channels cannot be divided by number of offsets"
        assert offset_plus is None or offset_plus.size(1) == self.offset.size(0)

        out_LO = None

        if not self.disable_displace:
            chan_per_pos = num_channels // free_channels

            if offset_plus is not None:
                offset = self.offset[None] + offset_plus
            else:
                offset = self.offset

            if self.regress_offset:
                assert offset_plus is None
                assert self.offset.dim() == 2
                offset = self.offset[None] + self.offset_regressor(inp).mean(dim=-1).mean(dim=-1).view(batch_size, free_channels, 2)

            if offset.dim() == 3:
                out = DisplaceCUDA.apply(inp.view(1, -1, height, width), offset.detach().round().int().view(-1, 2), chan_per_pos).view(batch_size, -1, height, width)
            elif offset.dim() == 2:
                out = DisplaceCUDA.apply(inp, offset.detach().round().int(), chan_per_pos)
            else:
                raise ValueError()

            if self.learnable_offset and (LO_active is True or (LO_active is None and self.LO_active is True)):
                if device not in self.field:
                    self.field[device] = self.field[torch.device("cpu")].to(device)

                out_LO = self.gaussian_translate(out, offset)
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
