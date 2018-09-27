import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars
from utils.log import log_i
import torch.nn.functional as F
import numpy as np
from .displace import Displace, DisplaceCUDA

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, init_stride,
                 fill=False, learnable_offset=False, LO_kernel_size=3, LO_sigma=0.5,
                 disable_displace=False, random_offset=0, use_origin=False, actual_stride=None,
                 displace_size=None, LO_balance_grad=True, free_chan_per_pos=1,
                 dconv_for_LO_stride=1):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.init_stride = init_stride
        self.fill = fill
        self.learnable_offset = learnable_offset
        self.disable_displace = disable_displace
        self.random_offset = random_offset
        self.use_origin = use_origin
        self.actual_stride = actual_stride
        self.displace_size = displace_size
        self.free_chan_per_pos = free_chan_per_pos
        self.dconv_for_LO_stride = dconv_for_LO_stride
        self.num_y, self.num_x, self.num_pos = self.get_num_offset(height, width, displace_size, init_stride, fill, use_origin)

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
            else:
                self.switch_LO_state(False)

    def set_learnable_offset_para(self, kernel_size, sigma):
        # TODO: Log
        log_i("learnable offset para set to kernel_size={}, sigma={}".format(kernel_size, sigma))
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
        if not self.fill:
            area = (self.width - self.offset.data[:, 0].round().abs()) * (self.height - self.offset.data[:, 1].round().abs())
            return grad / area.view(-1, 1)
        else:
            return grad / self.width / self.height

    @staticmethod
    def get_num_offset(height, width, displace_size, init_stride, fill, use_origin):
        if displace_size:
            num_x = displace_size[0]
            num_y = displace_size[1]
        else:
            assert int(height) - init_stride > init_stride
            assert int(width) - init_stride > init_stride
            if not fill:
                num_y = (int(height) - init_stride) // init_stride * 2 + 1
                num_x = (int(width) - init_stride) // init_stride * 2 + 1
            else:
                num_y = (int(height) - init_stride) // init_stride + 1
                num_x = (int(width) - init_stride) // init_stride + 1

        num_pos = num_y * num_x
        if not use_origin:
            num_pos -= 1
        return num_y, num_x, num_pos

    def init_offset(self):
        nh, nw = self.num_y, self.num_x
            
        if not self.fill:
            if self.random_offset > 0:
                self.offset.data.uniform_(-self.random_offset, self.random_offset)
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
        else:
            if self.random_offset is not None and self.random_offset > 0:
                self.offset.data.uniform_(0, self.random_offset)
                return
            count_off = 0
            for ih in list(range(0, nh // 2 + 1)) + list(range(-nh // 2, 0)) :
                for iw in list(range(0, nw // 2 + 1)) + list(range(-nw // 2, 0)):
                    if not self.use_origin and ih == 0 and iw == 0:
                        continue
                    if not self.actual_stride:
                        iwa = nw + iw if iw < 0 else iw
                        iha = nh + ih if ih < 0 else ih
                        self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 0] = iwa * self.init_stride
                        self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 1] = iha * self.init_stride
                    else:
                        self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 0] = iw * self.actual_stride if iw > 0 else self.width + iw * self.actual_stride
                        self.offset.data[count_off*self.free_chan_per_pos:(count_off+1)*self.free_chan_per_pos, 1] = ih * self.actual_stride if ih > 0 else self.height + ih * self.actual_stride
                    count_off += 1

    def reset_outsider(self):
        if not self.fill:
            self.offset.data[:, 0].clamp_(min=-self.width + 0.6,
                                          max=self.width - 0.6)
            self.offset.data[:, 1].clamp_(min=-self.height + 0.6,
                                          max=self.height - 0.6)
        else:
            off_x_rounded = self.offset.data[:, 0].round()
            off_y_rounded = self.offset.data[:, 1].round()
            self.offset.data[:, 0] -= (off_x_rounded / float(self.width)).floor() * float(self.width)
            self.offset.data[:, 1] -= (off_y_rounded / float(self.height)).floor() * float(self.height)

    def forward(self, inp, LO_active=None):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        device = inp.device
        assert self.height == height and self.width == width
        assert num_channels % (self.num_pos * self.free_chan_per_pos) == 0, "num of channels cannot be divided by number of positions"

        out_LO = None

        if not self.disable_displace:
            chan_per_pos = num_channels // (self.num_pos * self.free_chan_per_pos)
            if self.fill:
                inpinp = inp.repeat(1, 1, 2, 2)
                out = Displace.apply(inpinp, self.offset.detach().round().int(), chan_per_pos, True)
            else:
                out = DisplaceCUDA.apply(inp, self.offset.detach().round().int(), chan_per_pos)

            # out: nsamp x npos*chan_per_pos x height x width

            if self.learnable_offset and (LO_active is True or (LO_active is None and self.LO_active is True)):
                if device not in self.field:
                    self.field[device] = self.field[torch.device("cpu")].to(device)
                field = self.field[device]
                # npos x kh x kw
                # The offset means which direction and how long we should move the image,
                # while for each kernel at each individual place, -offset is the center where it should 'look at',
                # so the field becomes to force - (-offset.float) -> force + offset.float
                kernel = torch.exp(- (field + (self.offset - self.offset.detach().round())[:, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
                kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
                assert self.LO_kernel_size % 2 == 1
                # npos*chan_per_pos x kh x kw
                kernel = kernel.view(self.num_pos*self.free_chan_per_pos, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, chan_per_pos, 1, 1).view(num_channels, 1, self.LO_kernel_size, self.LO_kernel_size)
                # nsamp x npos*chan_per_pos x height x width
                out_LO = F.conv2d(out, kernel, None, (1, 1), (self.LO_kernel_size // 2 * self.dconv_for_LO_stride, self.LO_kernel_size // 2 * self.dconv_for_LO_stride), (self.dconv_for_LO_stride, self.dconv_for_LO_stride), num_channels)
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
