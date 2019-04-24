import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from torch._thnn import type2backend
import numpy as np

from torch.nn.modules.utils import _pair

from .src.jit import get_module
print("JIT building displace_cuda ...")
displace_cuda = get_module()

__all__ = ["DisplaceCUDA", "Displace", "DisplaceFracCUDA", "CustomizedGradConv2dCUDNN", "CustomizedGradDepthwiseConv2d", "CustomizedGradConv2dCUDNN2", "CustomizedGradDepthwiseConv2d2", "PositionalDisplace", "PositionalGaussianDisplace"]

class CustomizedGradConv2dCUDNN(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, offsets=None, chan_per_offset=1, side_thickness=1):
        assert torch.backends.cudnn.enabled
        ctx.save_for_backward(inp, weight)
        ctx._backend = type2backend[inp.type()]
        ctx.offsets = offsets
        ctx.chan_per_offset = chan_per_offset
        ctx.side_thickness = _pair(side_thickness)
        ctx.have_bias = (bias is not None)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        output = torch.cudnn_convolution(inp, weight, bias, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = displace_cuda.cudnn_convolution_backward_input(tuple(inp.size()), grad_output, weight, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

        if ctx.offsets is not None:
            new_grad_output = torch.empty_like(grad_output)
            displace_cuda.offset_mask(ctx._backend.library_state, grad_output, ctx.offsets, ctx.chan_per_offset, new_grad_output, ctx.side_thickness)
            grad_output = new_grad_output

        grad_weight = displace_cuda.cudnn_convolution_backward_weight(weight.size(), grad_output, inp, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        if ctx.have_bias:
            grad_bias = displace_cuda.cudnn_convolution_backward_bias(grad_output)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class CustomizedGradDepthwiseConv2d(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, offsets=None, chan_per_offset=1, side_thickness=1):
        assert groups == inp.size(1)
        ctx.save_for_backward(inp, weight)
        ctx._backend = type2backend[inp.type()]
        ctx.offsets = offsets
        ctx.chan_per_offset = chan_per_offset
        ctx.side_thickness = _pair(side_thickness)
        ctx.have_bias = (bias is not None)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.dilation = _pair(dilation)
        output = torch.empty_like(inp)
        ctx._backend.SpatialDepthwiseConvolution_updateOutput(
            ctx._backend.library_state,
            inp,
            output,
            weight,
            bias,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = torch.empty_like(inp)
        ctx._backend.SpatialDepthwiseConvolution_updateGradInput(
            ctx._backend.library_state,
            inp,
            grad_output,
            grad_input,
            weight,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])

        if ctx.offsets is not None:
            new_grad_output = torch.empty_like(grad_output)
            displace_cuda.offset_mask(ctx._backend.library_state, grad_output, ctx.offsets, ctx.chan_per_offset, new_grad_output, ctx.side_thickness)
            grad_output = new_grad_output

        grad_weight = torch.zeros_like(weight)
        ctx._backend.SpatialDepthwiseConvolution_accGradParameters(
            ctx._backend.library_state,
            inp,
            grad_output,
            grad_weight,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])

        if ctx.have_bias:
            grad_bias = grad_output.sum(0).sum(1).view(-1)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

# The following two function use offset_mask_frac function instead of offset_mask
class CustomizedGradConv2dCUDNN2(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, offsets=None, chan_per_offset=1):
        assert torch.backends.cudnn.enabled
        ctx.save_for_backward(inp, weight)
        ctx._backend = type2backend[inp.type()]
        ctx.offsets = offsets
        ctx.chan_per_offset = chan_per_offset
        ctx.have_bias = (bias is not None)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        output = torch.cudnn_convolution(inp, weight, bias, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = displace_cuda.cudnn_convolution_backward_input(tuple(inp.size()), grad_output, weight, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

        if ctx.offsets is not None:
            new_grad_output = torch.empty_like(grad_output)
            displace_cuda.offset_mask_frac(ctx._backend.library_state, grad_output, ctx.offsets, ctx.chan_per_offset, new_grad_output)
            grad_output = new_grad_output

        grad_weight = displace_cuda.cudnn_convolution_backward_weight(weight.size(), grad_output, inp, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        if ctx.have_bias:
            grad_bias = displace_cuda.cudnn_convolution_backward_bias(grad_output)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class CustomizedGradDepthwiseConv2d2(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, offsets=None, chan_per_offset=1):
        assert groups == inp.size(1)
        ctx.save_for_backward(inp, weight)
        ctx._backend = type2backend[inp.type()]
        ctx.offsets = offsets
        ctx.chan_per_offset = chan_per_offset
        ctx.have_bias = (bias is not None)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.dilation = _pair(dilation)
        output = torch.empty_like(inp)
        ctx._backend.SpatialDepthwiseConvolution_updateOutput(
            ctx._backend.library_state,
            inp,
            output,
            weight,
            bias,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_input = torch.empty_like(inp)
        ctx._backend.SpatialDepthwiseConvolution_updateGradInput(
            ctx._backend.library_state,
            inp,
            grad_output,
            grad_input,
            weight,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])

        if ctx.offsets is not None:
            new_grad_output = torch.empty_like(grad_output)
            displace_cuda.offset_mask_frac(ctx._backend.library_state, grad_output, ctx.offsets, ctx.chan_per_offset, new_grad_output)
            grad_output = new_grad_output

        grad_weight = torch.zeros_like(weight)
        ctx._backend.SpatialDepthwiseConvolution_accGradParameters(
            ctx._backend.library_state,
            inp,
            grad_output,
            grad_weight,
            weight.size(3), weight.size(2),
            ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0],
            ctx.dilation[1], ctx.dilation[0])

        if ctx.have_bias:
            grad_bias = grad_output.sum(0).sum(1).view(-1)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class DisplaceFracCUDA(Function):
    @staticmethod
    def forward(ctx, inp, offsets, chan_per_pos):
        ctx.chan_per_pos = chan_per_pos
        ctx._backend = type2backend[inp.type()]
        ctx.save_for_backward(inp, offsets)
        out = torch.empty_like(inp)
        displace_cuda.displace_frac_forward(ctx._backend.library_state,
            inp, offsets, chan_per_pos, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, offsets = ctx.saved_tensors
        grad_inp = torch.empty_like(grad_out)
        grad_offsets = torch.zeros_like(offsets)
        displace_cuda.displace_frac_backward(ctx._backend.library_state,
            grad_inp, offsets, ctx.chan_per_pos, grad_out)
        displace_cuda.displace_frac_offset_backward(ctx._backend.library_state,
            inp, offsets, grad_offsets, ctx.chan_per_pos, grad_out)

        return grad_inp, grad_offsets, None

class DisplaceCUDA(Function):
    @staticmethod
    def forward(ctx, inp, offsets, chan_per_pos):
        ctx.chan_per_pos = chan_per_pos
        ctx._backend = type2backend[inp.type()]
        ctx.save_for_backward(offsets)
        # empty_like could cause test_displace inaccurate
        # following functions could reuse previous result
        out = torch.empty_like(inp)
        displace_cuda.displace_forward(ctx._backend.library_state,
            inp, offsets, chan_per_pos, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (offsets,) = ctx.saved_tensors
        # empty_like could cause test_displace inaccurate
        # following functions could reuse previous result
        grad_inp = torch.empty_like(grad_out)
        displace_cuda.displace_backward(ctx._backend.library_state,
            grad_inp, offsets, ctx.chan_per_pos, grad_out)

        return grad_inp, None, None

class Displace(Function):
    @staticmethod
    def forward(ctx, inp, offsets, chan_per_pos, fill=False):
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

class PositionalDisplace(Function):
    @staticmethod
    def forward(ctx, inp, offsets_x, offsets_y, chan_per_pos, blur_kernel=None):
        ctx.chan_per_pos = chan_per_pos
        ctx._backend = type2backend[inp.type()]
        ctx.save_for_backward(inp, offsets_x, offsets_y, blur_kernel)
        out = torch.empty_like(inp)
        displace_cuda.displace_pos_sep_forward(ctx._backend.library_state,
            inp, offsets_x, offsets_y, chan_per_pos, out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, offsets_x, offsets_y, blur_kernel = ctx.saved_tensors
        grad_inp = torch.zeros_like(grad_out)
        grad_offsets_x = torch.zeros_like(offsets_x)
        grad_offsets_y = torch.zeros_like(offsets_y)
        if blur_kernel is None:
            displace_cuda.displace_pos_sep_backward(ctx._backend.library_state,
                inp, grad_inp, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, ctx.chan_per_pos, grad_out)
        else:
            displace_cuda.displace_pos_sep_backward_data(ctx._backend.library_state,
                grad_inp, offsets_x, offsets_y, ctx.chan_per_pos, grad_out)
            inp_blur = F.conv2d(inp, blur_kernel, padding=(blur_kernel.size(2) // 2, blur_kernel.size(3) // 2), groups=blur_kernel.size(0))
            # grad_out_blur = F.conv2d(grad_out, blur_kernel, padding=(blur_kernel.size(2) // 2, blur_kernel.size(3) // 2), groups=blur_kernel.size(0))
            displace_cuda.displace_pos_sep_backward_offset(ctx._backend.library_state,
                inp_blur, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, ctx.chan_per_pos, grad_out)

        return grad_inp, grad_offsets_x, grad_offsets_y, None, None

class PositionalGaussianDisplace(Function):
    @staticmethod
    def forward(ctx, inp, offsets_x, offsets_y, channel_per_off, angles, scales, gaus_weight, fill=0, simple=False):
        ctx._backend = type2backend[inp.type()]
        ctx.channel_per_off = channel_per_off
        ctx.fill = fill
        ctx.simple = simple
        out = torch.zeros_like(inp)
        if not simple:
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            displace_cuda.displace_gaus_forward(ctx._backend.library_state,
                inp, offsets_x, offsets_y, channel_per_off, out, angles, scales, gaus_weight, cos_angles, sin_angles, fill)
            ctx.save_for_backward(inp, offsets_x, offsets_y, angles, scales, gaus_weight, cos_angles, sin_angles)
        else:
            assert fill == 0
            offsets_x_rounded = offsets_x.round().int()
            offsets_y_rounded = offsets_y.round().int()
            displace_cuda.displace_pos_sep_forward(ctx._backend.library_state,
                inp, offsets_x_rounded, offsets_y_rounded, channel_per_off, out)
            ctx.save_for_backward(inp, offsets_x, offsets_y, angles, scales, gaus_weight, offsets_x_rounded, offsets_y_rounded)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        inp, offsets_x, offsets_y, angles, scales, gaus_weight = ctx.saved_tensors[:6]
        grad_inp = torch.zeros_like(inp)
        if gaus_weight.requires_grad:
            grad_gaus_weight = torch.zeros_like(gaus_weight)
        else:
            grad_gaus_weight = None
        if offsets_x.requires_grad or offsets_y.requires_grad:
            grad_offsets_x = torch.zeros_like(offsets_x)
            grad_offsets_y = torch.zeros_like(offsets_y)
        else:
            grad_offsets_x = None
            grad_offsets_y = None

        if not ctx.simple:
            cos_angles, sin_angles = ctx.saved_tensors[6:]

            displace_cuda.displace_gaus_backward(ctx._backend.library_state,
                inp, grad_inp, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, ctx.channel_per_off, grad_out,
                angles, scales, gaus_weight, grad_gaus_weight, cos_angles, sin_angles, ctx.fill, ctx.simple)
        else:
            offsets_x_rounded, offsets_y_rounded = ctx.saved_tensors[6:]
            displace_cuda.displace_pos_sep_backward(ctx._backend.library_state,
                None, grad_inp, offsets_x_rounded, offsets_y_rounded, None, None, ctx.channel_per_off, grad_out)
            if not (grad_gaus_weight is None and grad_offsets_x is None and grad_offsets_y is None):
                cos_angles = torch.cos(angles)
                sin_angles = torch.sin(angles)
                displace_cuda.displace_gaus_backward(ctx._backend.library_state,
                    inp, None, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, ctx.channel_per_off, grad_out,
                    angles, scales, gaus_weight, grad_gaus_weight, cos_angles, sin_angles, ctx.fill, ctx.simple)

        return grad_inp, grad_offsets_x, grad_offsets_y, None, None, None, grad_gaus_weight, None, None, None