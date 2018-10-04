import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from torch._thnn import type2backend

from torch.nn.modules.utils import _pair
from . import displace_cuda
# These functions are not exported in pytorch python bindings, so we borrow displace_cuda to export them
from .displace_cuda import cudnn_convolution_backward_input, cudnn_convolution_backward_weight, cudnn_convolution_backward_bias

__all__ = ["DisplaceCUDA", "Displace", "CustomizedGradConv2dCUDNN", "CustomizedGradDepthwiseConv2d"]

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
        grad_input = cudnn_convolution_backward_input(tuple(inp.size()), grad_output, weight, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)

        if ctx.offsets is not None:
            new_grad_output = torch.empty_like(grad_output)
            displace_cuda.offset_mask(ctx._backend.library_state, grad_output, ctx.offsets, ctx.chan_per_offset, new_grad_output, ctx.side_thickness)
            grad_output = new_grad_output

        grad_weight = cudnn_convolution_backward_weight(weight.size(), grad_output, inp, ctx.padding, ctx.stride, ctx.dilation, ctx.groups, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        if ctx.have_bias:
            grad_bias = cudnn_convolution_backward_bias(grad_output)
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
            grad_bias = grad_output.contiguous().view(grad_output.size(0), grad_output.size(1), -1).sum(0).sum(1)
        else:
            grad_bias = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

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
