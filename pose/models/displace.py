import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars
import torch.nn.functional as F
import numpy as np

class Displace(Function):
    @staticmethod
    def forward(ctx, inp, offsets, chan_per_pos, fill):
        ctx.fill = fill
        ctx.chan_per_pos = chan_per_pos

        if not fill:
            out = torch.zeros_like(inp)
            height = inp.size(2)
            width = inp.size(3)
            lefttop_out = offsets.clamp(min=0)
            lefttop_inp = (-offsets).clamp(min=0)
            rightbot_out = offsets.clamp(max=0) + torch.tensor([[width, height]], dtype=offsets.dtype, device=offsets.device)
            rightbot_inp = lefttop_inp + (rightbot_out - lefttop_out)
            ctx.save_for_backward(lefttop_out, lefttop_inp, rightbot_out, rightbot_inp)

            cs = 0
            for i in range(offsets.size(0)):
                out[:, cs:cs+chan_per_pos, lefttop_out[i,1]:rightbot_out[i,1], lefttop_out[i,0]:rightbot_out[i,0]] = \
                    inp[:, cs:cs+chan_per_pos, lefttop_inp[i,1]:rightbot_inp[i,1], lefttop_inp[i,0]:rightbot_inp[i,0]]
                cs += chan_per_pos
        else:
            inphw = inp.size()[2:]
            height_out = inphw[0] // 2
            width_out = inphw[1] // 2
            ctx.save_for_backward(offsets)

            out = torch.zeros(inp.size()[:2] + (height_out, width_out), dtype=inp.dtype, device=inp.device)
            cs = 0
            for i in range(offsets.size(0)):
                out[:, cs:cs+chan_per_pos] = \
                    inp[:, cs:cs+chan_per_pos, offsets[i,1]:offsets[i,1]+height_out, offsets[i,0]:offsets[i,0]+width_out]
                cs += chan_per_pos

        return out

    @staticmethod
    def backward(ctx, grad_out):
        if not ctx.fill:
            (lefttop_out, lefttop_inp, rightbot_out, rightbot_inp) = ctx.saved_tensors
            grad_inp = torch.zeros_like(grad_out)
            cs = 0
            for i in range(lefttop_out.size(0)):
                grad_inp[:, cs:cs+ctx.chan_per_pos, lefttop_inp[i,1]:rightbot_inp[i,1], lefttop_inp[i,0]:rightbot_inp[i,0]] = \
                    grad_out[:, cs:cs+ctx.chan_per_pos, lefttop_out[i,1]:rightbot_out[i,1], lefttop_out[i,0]:rightbot_out[i,0]]
                cs += ctx.chan_per_pos
        else:
            (offsets,) = ctx.saved_tensors
            height_out = grad_out.size(2)
            width_out = grad_out.size(3)
            grad_inp = torch.zeros(grad_out.size()[:2] + (height_out*2, width_out*2), dtype=grad_out.dtype, device=grad_out.device)
            cs = 0
            for i in range(offsets.size(0)):
                grad_inp[:, cs:cs+ctx.chan_per_pos, offsets[i,1]:offsets[i,1]+height_out, offsets[i,0]:offsets[i,0]+width_out] = \
                    grad_out[:, cs:cs+ctx.chan_per_pos]
                cs += ctx.chan_per_pos

        return grad_inp, None, None, None

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, init_stride,
                 fill=False, learnable_offset=False, LO_kernel_size=3, LO_sigma=0.5,
                 disable_displace=False, random_offset=0, use_origin=False):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.init_stride = init_stride
        self.fill = fill
        self.learnable_offset = learnable_offset
        self.disable_displace = disable_displace
        self.random_offset = random_offset
        self.use_origin = use_origin
        if not fill:
            self.num_y = (height - init_stride) // init_stride * 2 + 1
            self.num_x = (width - init_stride) // init_stride * 2 + 1
        else:
            self.num_y = (height - init_stride) // init_stride + 1
            self.num_x = (width - init_stride) // init_stride + 1
        self.num_pos = self.num_y * self.num_x
        if not use_origin:
            self.num_pos -= 1

        if not disable_displace:
            self.offset = nn.parameter.Parameter(torch.Tensor(self.num_pos, 2), requires_grad=False)
            self.init_offset()
            if learnable_offset:
                assert isinstance(LO_kernel_size, int)
                assert LO_kernel_size % 2 == 1

                self.LO_kernel_size = LO_kernel_size
                self.LO_sigma = LO_sigma
                self.offset.requires_grad = True
                self.offset.register_hook(self.balance_offset_grad)

                x = torch.arange(LO_kernel_size, dtype=torch.float).view(1, -1, 1).expand(LO_kernel_size, -1, -1) - float(LO_kernel_size // 2)
                y = torch.arange(LO_kernel_size, dtype=torch.float).view(-1, 1, 1).expand(-1, LO_kernel_size, -1) - float(LO_kernel_size // 2)
                self.field = dict()
                self.field[torch.device("cpu")] = torch.cat([x, y], dim=2).expand(1, -1, -1, -1).repeat(self.num_pos, 1, 1, 1)

    def balance_offset_grad(self, grad):
        if not self.fill:
            area = (self.width - self.offset.data[:, 0].round().abs()) * (self.height - self.offset.data[:, 1].round().abs())
            return grad / area.view(-1, 1)
        else:
            return grad / self.width / self.height

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
                    self.offset.data[count_off, 0] = iw * self.init_stride
                    self.offset.data[count_off, 1] = ih * self.init_stride
                    count_off += 1
        else:
            if self.random_offset is not None and self.random_offset > 0:
                self.offset.data.uniform_(0, self.random_offset)
                return
            count_off = 0
            for ih in range(0, nh):
                for iw in range(0, nw):
                    if not self.use_origin and ih == 0 and iw == 0:
                        continue
                    self.offset.data[count_off, 0] = iw * self.init_stride
                    self.offset.data[count_off, 1] = ih * self.init_stride
                    count_off += 1

    def reset_outsider(self):
        if not self.fill:
            self.offset.data[:, 0].clamp_(min=-self.width + 0.5 + np.finfo(np.float32).eps.item(),
                                          max=self.width - 0.5 - np.finfo(np.float32).eps.item())
            self.offset.data[:, 1].clamp_(min=-self.height + 0.5 + np.finfo(np.float32).eps.item(),
                                          max=self.height - 0.5 - np.finfo(np.float32).eps.item())
        else:
            off_x_rounded = self.offset.data[:, 0].round()
            off_y_rounded = self.offset.data[:, 1].round()
            self.offset.data[:, 0] -= (off_x_rounded / float(self.width)).floor() * float(self.width)
            self.offset.data[:, 1] -= (off_y_rounded / float(self.height)).floor() * float(self.height)

    def forward(self, inp):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        device = inp.device
        assert self.height == height and self.width == width
        assert num_channels % self.num_pos == 0, "num of channels cannot be divided by number of positions"

        if not self.disable_displace:
            if device not in self.field:
                self.field[device] = self.field[torch.device("cpu")].to(device)
                assert len(self.field.keys()) < 10
            field = self.field[device]

            chan_per_pos = num_channels // self.num_pos
            if self.fill:
                inpinp = inp.repeat(1, 1, 2, 2)
                out = Displace.apply(inpinp, self.offset.detach().round().int(), chan_per_pos, True)
            else:
                out = Displace.apply(inp, self.offset.detach().round().int(), chan_per_pos, False)

            # out: nsamp x npos*chan_per_pos x height x width

            if self.learnable_offset:
                # npos x kh x kw
                # The offset means which direction and how long we should move the image,
                # while for each kernel at each individual place, -offset is the center where it should 'look at',
                # so the field becomes to force - (-offset.float) -> force + offset.float
                kernel = torch.exp(- (field + (self.offset - self.offset.detach().round())[:, None, None, :]).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
                kernel = kernel / kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
                # npos*chan_per_pos x kh x kw
                kernel = kernel.view(self.num_pos, 1, self.LO_kernel_size, self.LO_kernel_size).repeat(1, chan_per_pos, 1, 1).view(self.num_pos*chan_per_pos, 1, self.LO_kernel_size, self.LO_kernel_size)
                # nsamp x npos*chan_per_pos x height x width
                out = F.conv2d(out, kernel, None, (1, 1), (self.LO_kernel_size // 2, self.LO_kernel_size // 2), (1, 1), self.num_pos*chan_per_pos)
        else:
            out = inp

        if config.vis:
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

        return out