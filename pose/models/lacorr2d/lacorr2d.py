import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

from . import lacorr2d_cuda

class LocalAutoCorr2DCUDAFunction(Function):
    @staticmethod
    def forward(ctx, inp, kernel_size, stride_size):
        ctx.save_for_backward(inp)
        ctx.kernel_size = kernel_size
        ctx.stride_size = stride_size
        return lacorr2d_cuda.lacorr2d_forward(inp, kernel_size[1], kernel_size[0], stride_size[1], stride_size[0])[0]

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_inp = lacorr2d_cuda.lacorr2d_backward(inp, grad_output, ctx.kernel_size[1], ctx.kernel_size[0], ctx.stride_size[1], ctx.stride_size[0])[0]
        return grad_inp, None, None

class LocalAutoCorr2DCUDA(Module):
    def __init__(self, kernel_size, stride_size):
        super(LocalAutoCorr2DCUDA, self).__init__()
        self.kernel_size = kernel_size
        self.stride_size = stride_size

    def forward(self, x):
        return LocalAutoCorr2DCUDAFunction.apply(x, self.kernel_size, self.stride_size)

class LocalAutoCorr2D(Module):
    def __init__(self, kernel_size, stride_size):
        super(LocalAutoCorr2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride_size = stride_size

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        height = x.size(2)
        width = x.size(3)

        n_corr_w = (width - self.kernel_size[0]) // self.stride_size[0] + 1
        n_corr_h = (height - self.kernel_size[1]) // self.stride_size[1] + 1
        n_corr = n_corr_w * n_corr_h
        corr_size = self.kernel_size[0] * self.kernel_size[1]

        all_weight = list()
        all_bg = list()

        for i_corr_y in range(n_corr_h):
            for i_corr_x in range(n_corr_w):
                left = i_corr_x * self.stride_size[0]
                right = left + self.kernel_size[0]
                top = i_corr_y * self.stride_size[1]
                bottom = top + self.kernel_size[1]
                left_bg = left - self.kernel_size[0] // 2
                right_bg = right + (self.kernel_size[0] + 1) // 2 - 1
                top_bg = top - self.kernel_size[1] // 2
                bottom_bg = bottom + (self.kernel_size[1] + 1) // 2 - 1

                weight = x[:, :, top:bottom, left:right]
                bg = x[:, :, max(0, top_bg):min(height, bottom_bg), max(0, left_bg):min(width, right_bg)]
                if top_bg < 0:
                    bg = torch.cat([torch.zeros((bg.size(0), bg.size(1), -top_bg, bg.size(3)), dtype=x.dtype, device=x.device), bg], dim=2)
                if bottom_bg > height:
                    bg = torch.cat([bg, torch.zeros((bg.size(0), bg.size(1), bottom_bg - height, bg.size(3)), dtype=x.dtype, device=x.device)], dim=2)
                if left_bg < 0:
                    bg = torch.cat([torch.zeros((bg.size(0), bg.size(1), bg.size(2), -left_bg), dtype=x.dtype, device=x.device), bg], dim=3)
                if right_bg > width:
                    bg = torch.cat([bg, torch.zeros((bg.size(0), bg.size(1), bg.size(2), right_bg - width), dtype=x.dtype, device=x.device)], dim=3)

                all_weight.append(weight.view(weight.size(0), 1, weight.size(1), weight.size(2), weight.size(3)))
                all_bg.append(bg.view(bg.size(0), 1, bg.size(1), bg.size(2), bg.size(3)))

        all_weight = torch.cat(all_weight, dim=1)
        all_weight = all_weight.view(-1, 1, all_weight.size(3), all_weight.size(4))
        all_bg = torch.cat(all_bg, dim=1)
        all_bg = all_bg.view(1, -1, all_bg.size(3), all_bg.size(4))

        # nbat*n_corr_h*n_corr_w*nchannel x 1 x kh x kw
        ret = F.conv2d(all_bg, all_weight, None, groups=all_weight.size(0))
        ret = ret.view(batch_size, n_corr_h, n_corr_w, channel_size, self.kernel_size[1], self.kernel_size[0])

        return ret