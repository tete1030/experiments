import torch
from torch import nn
from torch.autograd import Function
from utils.globals import config, globalvars

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
    def __init__(self, stride):
        super(DisplaceChannel, self).__init__()
        self.stride = stride

    @staticmethod
    def calc_num_positions(height, width, stride):
        nh = (height - stride) // stride * 2 + 1
        nw = (width - stride) // stride * 2 + 1

        return nh, nw, nh * nw - 1

    def forward(self, x):
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        nh, nw, num_pos = self.calc_num_positions(height, width, self.stride)
        chan_per_pos = num_channels // num_pos
        assert num_channels % num_pos == 0, "num of channels cannot be divided by number of positions"
        count_chan = 0
        channels = []
        for ih in range(-(nh // 2), nh // 2 + 1):
            for iw in range(-(nw // 2), nw // 2 + 1):
                if ih == 0 and iw == 0:
                    continue
                channels.append(Displace.apply(x[:, count_chan:count_chan+chan_per_pos], iw * self.stride, ih * self.stride))
                count_chan += chan_per_pos

        out = torch.cat(channels, dim=1)

        if config.vis:
            import matplotlib.pyplot as plt
            import cv2
            fig, axes = plt.subplots(3, 30, figsize=(100, 12), squeeze=False)
            
            for row, axes_row in enumerate(axes):
                img = (globalvars.cur_img.data[row].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
                fts = x.data[row].cpu().numpy()
                for col, ax in enumerate(axes_row):
                    if col == 0:
                        ax.imshow(img)
                    else:
                        ax.imshow(fts[col-1])
            fig.suptitle("displace x")

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