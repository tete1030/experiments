import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars
import torch.nn.functional as F

class Displace(Function):
    @staticmethod
    def forward(ctx, inp, off_x, off_y):
        ctx.top_out = max(0, off_y)
        ctx.top_inp = max(0, -off_y)
        ctx.bottom_out = inp.size(2) + min(0, off_y)
        ctx.bottom_inp = ctx.top_inp + ctx.bottom_out - ctx.top_out
        ctx.left_out = max(0, off_x)
        ctx.left_inp = max(0, -off_x)
        ctx.right_out = inp.size(3) + min(0, off_x)
        ctx.right_inp = ctx.left_inp + ctx.right_out - ctx.left_out

        out = torch.zeros_like(inp, requires_grad=False)
        out[:, :, ctx.top_out:ctx.bottom_out, ctx.left_out:ctx.right_out] = \
            inp[:, :, ctx.top_inp:ctx.bottom_inp, ctx.left_inp:ctx.right_inp]

        return out

    @staticmethod
    def backward(ctx, grad_out):
        grad_inp = torch.zeros_like(grad_out, requires_grad=False)
        grad_inp[:, :, ctx.top_inp:ctx.bottom_inp, ctx.left_inp:ctx.right_inp] = \
            grad_out[:, :, ctx.top_out:ctx.bottom_out, ctx.left_out:ctx.right_out]

        return grad_inp, None, None

class DisplaceChannel(nn.Module):
    def __init__(self, height, width, init_stride, fill=False, learnable_offset=False, LO_kernel_size=3, LO_sigma=0.5):
        super(DisplaceChannel, self).__init__()
        self.height = height
        self.width = width
        self.init_stride = init_stride
        self.fill = fill
        self.learnable_offset = learnable_offset
        if not fill:
            self.num_y = (height - init_stride) // init_stride * 2 + 1
            self.num_x = (width - init_stride) // init_stride * 2 + 1
        else:
            self.num_y = (height - init_stride) // init_stride + 1
            self.num_x = (width - init_stride) // init_stride + 1
        self.num_pos = self.num_y * self.num_x - 1
        self.offset = nn.parameter.Parameter(torch.Tensor(self.num_pos, 2), requires_grad=False)
        self.init_offset()
        if learnable_offset:
            assert isinstance(LO_kernel_size, int)
            assert LO_kernel_size % 2 == 1

            self.LO_kernel_size = LO_kernel_size
            self.LO_sigma = LO_sigma
            self.offset.requires_grad = True

            x = torch.arange(LO_kernel_size, dtype=torch.float).view(1, -1, 1).expand(LO_kernel_size, -1, -1) - float(LO_kernel_size // 2)
            y = torch.arange(LO_kernel_size, dtype=torch.float).view(-1, 1, 1).expand(-1, LO_kernel_size, -1) - float(LO_kernel_size // 2)
            self.register_buffer("field", torch.cat([x, y], dim=2))

    def init_offset(self):
        nh, nw = self.num_y, self.num_x
        if not self.fill:
            count_off = 0
            for ih in range(-(nh // 2), nh // 2 + 1):
                for iw in range(-(nw // 2), nw // 2 + 1):
                    if ih == 0 and iw == 0:
                        continue
                    self.offset.data[count_off, 0] = iw * self.init_stride
                    self.offset.data[count_off, 1] = ih * self.init_stride
                    count_off += 1
        else:
            count_off = 0
            for ih in range(0, nh):
                for iw in range(0, nw):
                    if ih == 0 and iw == 0:
                        continue
                    self.offset.data[count_off, 0] = iw * self.init_stride
                    self.offset.data[count_off, 1] = ih * self.init_stride
                    count_off += 1

    def forward(self, inp):
        batch_size = inp.size(0)
        num_channels = inp.size(1)
        height = inp.size(2)
        width = inp.size(3)
        assert self.height == height and self.width == width
        assert num_channels % self.num_pos == 0, "num of channels cannot be divided by number of positions"

        chan_per_pos = num_channels // self.num_pos
        channels = []
        if self.fill:
            inpinp = inp.repeat(1, 1, 2, 2)
        for ipos in range(self.offset.size(0)):
            ichan = ipos * chan_per_pos
            off_x = self.offset[ipos, 0].detach().round().int().item()
            off_y = self.offset[ipos, 1].detach().round().int().item()
            # TODO: set limit on offset
            if not self.fill:
                offchannel = Displace.apply(inp[:, ichan:ichan+chan_per_pos], off_x, off_y)
            else:
                offchannel = inpinp[:, ichan:ichan+chan_per_pos, off_y:off_y+height, off_x:off_x+width].contiguous()
            if self.learnable_offset:
                kernel = torch.exp(- (self.field - (self.offset[ipos] - self.offset[ipos].detach().round()).expand(1, 1, -1)).pow(2).sum(dim=-1) / 2 / float(self.LO_sigma) ** 2)
                kernel = kernel / kernel.sum()
                kernel = kernel.expand(1, 1, -1, -1)
                offchannel = F.conv2d(offchannel.view(-1, 1, height, width), kernel, None, (1, 1), (self.LO_kernel_size // 2, self.LO_kernel_size // 2), (1, 1), 1)
                offchannel = offchannel.view(batch_size, -1, height, width)

            channels.append(offchannel)

        out = torch.cat(channels, dim=1)

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