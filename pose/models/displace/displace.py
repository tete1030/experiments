import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
from torch._thnn import type2backend
from . import displace_cuda

__all__ = ["DisplaceCUDA", "Displace"]

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
