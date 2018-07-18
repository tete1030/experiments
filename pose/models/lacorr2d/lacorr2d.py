import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function

from . import lacorr2d_cuda

class PadInfo(object):
    def __init__(self, top=0, bottom=0, left=0, right=0):
        assert isinstance(top, int) and isinstance(bottom, int) and isinstance(left, int) and isinstance(right, int)
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

class LocalAutoCorr2DCUDAFunction(Function):
    @staticmethod
    def forward(ctx, inp, kernel_size, stride_size, pad):
        ctx.save_for_backward(inp)
        ctx.kernel_size = kernel_size
        ctx.stride_size = stride_size
        ctx.pad = pad
        return lacorr2d_cuda.lacorr2d_forward(inp, kernel_size[0], kernel_size[1], stride_size[0], stride_size[1], pad.top, pad.bottom, pad.left, pad.right)[0]

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        grad_inp = lacorr2d_cuda.lacorr2d_backward(inp, grad_output, ctx.kernel_size[0], ctx.kernel_size[1], ctx.stride_size[0], ctx.stride_size[1], ctx.pad.top, ctx.pad.bottom, ctx.pad.left, ctx.pad.right)[0]
        return grad_inp, None, None, None

class LocalAutoCorr2DCUDA(Module):
    def __init__(self, kernel_size, stride_size, pad):
        super(LocalAutoCorr2DCUDA, self).__init__()
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.pad = pad

    def forward(self, x):
        return LocalAutoCorr2DCUDAFunction.apply(x, self.kernel_size, self.stride_size, self.pad)

class LocalAutoCorr2D(Module):
    def __init__(self, kernel_size, stride_size, pad):
        super(LocalAutoCorr2D, self).__init__()
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.pad = pad

    def forward(self, x):
        batch_size = x.size(0)
        channel_size = x.size(1)
        height = x.size(2)
        width = x.size(3)

        n_corr_w = (width + self.pad.left + self.pad.right - self.kernel_size[1]) // self.stride_size[1] + 1
        n_corr_h = (height + self.pad.top + self.pad.bottom - self.kernel_size[0]) // self.stride_size[0] + 1
        n_corr = n_corr_w * n_corr_h
        corr_size = self.kernel_size[0] * self.kernel_size[1]

        all_weight = list()
        all_bg = list()

        for i_corr_y in range(n_corr_h):
            for i_corr_x in range(n_corr_w):
                left = -self.pad.left + i_corr_x * self.stride_size[1]
                right = left + self.kernel_size[1]
                top = -self.pad.top + i_corr_y * self.stride_size[0]
                bottom = top + self.kernel_size[0]
                left_bg = left - self.kernel_size[1] // 2
                right_bg = right + (self.kernel_size[1] + 1) // 2 - 1
                top_bg = top - self.kernel_size[0] // 2
                bottom_bg = bottom + (self.kernel_size[0] + 1) // 2 - 1

                if ((top >= 0 and top < height) or (bottom > 0 and bottom <= height)) and \
                    ((left >= 0 and left < width) or (right > 0 and right <= width)):
                    weight = list()
                    width_inside = min(width, right) - max(0, left)
                    if top < 0:
                        weight.append(torch.zeros((x.size(0), x.size(1), -top + min(0, bottom), width_inside), dtype=x.dtype, device=x.device))
                    weight.append(x[:, :, max(0, top):min(height, bottom), max(0, left):min(width, right)])
                    if bottom > height:
                        weight.append(torch.zeros((x.size(0), x.size(1), bottom - height, width_inside), dtype=x.dtype, device=x.device))
                    weight = [torch.cat(weight, dim=2)]
                    if left < 0:
                        weight.insert(0, torch.zeros((x.size(0), x.size(1), self.kernel_size[0], -left + min(0, right)), dtype=x.dtype, device=x.device))
                    if right > width:
                        weight.append(torch.zeros((x.size(0), x.size(1), self.kernel_size[0], right - width), dtype=x.dtype, device=x.device))
                    weight = torch.cat(weight, dim=3)
                else:
                    weight = torch.zeros((x.size(0), x.size(1), self.kernel_size[0], self.kernel_size[1]), dtype=x.dtype, device=x.device)

                if ((top_bg >= 0 and top_bg < height) or (bottom_bg > 0 and bottom_bg <= height)) and \
                    ((left_bg >= 0 and left_bg < width) or (right_bg > 0 and right_bg <= width)):
                    bg = list()
                    width_inside = min(width, right_bg) - max(0, left_bg)
                    if top_bg < 0:
                        bg.append(torch.zeros((x.size(0), x.size(1), -top_bg + min(0, bottom_bg), width_inside), dtype=x.dtype, device=x.device))
                    bg.append(x[:, :, max(0, top_bg):min(height, bottom_bg), max(0, left_bg):min(width, right_bg)])
                    if bottom_bg > height:
                        bg.append(torch.zeros((x.size(0), x.size(1), bottom_bg - height, width_inside), dtype=x.dtype, device=x.device))
                    bg = [torch.cat(bg, dim=2)]
                    if left_bg < 0:
                        bg.insert(0, torch.zeros((x.size(0), x.size(1), bottom_bg - top_bg, -left_bg + min(0, right_bg)), dtype=x.dtype, device=x.device))
                    if right_bg > width:
                        bg.append(torch.zeros((x.size(0), x.size(1), bottom_bg - top_bg, right_bg - width), dtype=x.dtype, device=x.device))
                    bg = torch.cat(bg, dim=3)
                else:
                    bg = torch.zeros((x.size(0), x.size(1), bottom_bg - top_bg, right_bg - left_bg), dtype=x.dtype, device=x.device)

                all_weight.append(weight.view(weight.size(0), 1, weight.size(1), weight.size(2), weight.size(3)))
                all_bg.append(bg.view(bg.size(0), 1, bg.size(1), bg.size(2), bg.size(3)))

        all_weight = torch.cat(all_weight, dim=1)
        all_weight = all_weight.view(-1, 1, all_weight.size(3), all_weight.size(4))
        all_bg = torch.cat(all_bg, dim=1)
        all_bg = all_bg.view(1, -1, all_bg.size(3), all_bg.size(4))

        # nbat*n_corr_h*n_corr_w*nchannel x 1 x kh x kw
        ret = F.conv2d(all_bg, all_weight, None, groups=all_weight.size(0))
        ret = ret.view(batch_size, n_corr_h, n_corr_w, channel_size, self.kernel_size[0], self.kernel_size[1])

        return ret