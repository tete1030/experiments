#!python3
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import datetime
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from lib.models.displace import Displace, DisplaceCUDA, DisplaceFracCUDA, CustomizedGradDepthwiseConv2d2
from torch.autograd import Function

def zero_grad(t):
    if t.grad is not None:
        t.grad.zero_()

class _Model(nn.Module):
    def __init__(self, fn):
        super(_Model, self).__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn.apply(*args)

Model = lambda x: x if isinstance(x, nn.Module) else _Model(x)

class DisplaceFrac(nn.Module):
    def __init__(self, num_offsets):
        super(DisplaceFrac, self).__init__()
        x = torch.arange(3, dtype=torch.float).view(1, -1, 1).expand(3, -1, -1) - 1
        y = torch.arange(3, dtype=torch.float).view(-1, 1, 1).expand(-1, 3, -1) - 1
        field = torch.cat([x, y], dim=2).expand(1, -1, -1, -1)\
            .repeat(num_offsets, 1, 1, 1)
        self.register_buffer("field", field)

    # chan_per_pos is dummy
    def forward(self, x, offsets, chan_per_pos):
        assert chan_per_pos == 1
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        offsets_int = offsets.detach().round()
        offsets_frac = offsets - offsets_int
        offsets_int = offsets_int.int()
        x = DisplaceCUDA.apply(x, offsets_int, 1)
        kernel = (1 - (self.field + offsets_frac[:, None, None, :]).abs()).clamp(min=0).prod(dim=-1)
        kernel = kernel.view(-1, 1, 3, 3)
        x = CustomizedGradDepthwiseConv2d2.apply(
            x, kernel, None,
            (1, 1),
            (1, 1),
            (1, 1),
            num_channels,
            offsets,
            1)
        return x

def main(args):
    REPEAT_TIMES = args.repeat
    
    BATCH_SIZE = args.batch_size
    CHAN_PER_OFFSET = args.chan_per_offset
    NUM_OFFSET = args.num_offset
    CHANNEL_SIZE = CHAN_PER_OFFSET * NUM_OFFSET
    WIDTH = args.width
    HEIGHT = args.height
    if args.seed is None:
        torch.cuda.seed_all()
    else:
        print("Using seed {}".format(args.seed))
        torch.cuda.manual_seed_all(args.seed)
    # initial_seed will initialize CUDA
    print("seed = {}".format(torch.cuda.initial_seed()))

    functions = list()
    models = list()

    assert not (args.model == "all" and (args.memory or args.time))

    if args.model == "normal":
        functions.append(Displace if not args.frac else DisplaceFrac(CHANNEL_SIZE).cuda())
    elif args.model == "cuda":
        functions.append(DisplaceCUDA if not args.frac else DisplaceFracCUDA)
    elif args.model == "all":
        functions.extend([Displace, DisplaceCUDA] if not args.frac else [DisplaceFrac(CHANNEL_SIZE).cuda(), DisplaceFracCUDA])

    for fn in functions:
        models.append(Model(fn))

    if args.double:
        DTYPE = torch.float64
    else:
        DTYPE = torch.float32

    if args.thr == "0":
        threshold = 0.
    elif args.thr == "eps32":
        threshold = np.finfo(np.float32).eps.item()
    elif args.thr == "eps64":
        threshold = np.finfo(np.float64).eps.item()
    else:
        threshold = float(args.thr)

    if args.time:
        start_time = datetime.datetime.now()

    smax_output_error = 0.
    smax_grad_error = 0.
    if args.frac:
        smax_grad_off_error = 0.

    for i in range(REPEAT_TIMES):
        img = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda")
        if args.img_value is not None:
            img.fill_(args.img_value)
        label = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda")
        if args.label_value is not None:
            label.fill_(args.label_value)

        if not args.frac:
            offsets_w = torch.randint(low=-WIDTH, high=WIDTH, size=(NUM_OFFSET, 1), dtype=torch.int, device="cuda")
            offsets_h = torch.randint(low=-HEIGHT, high=HEIGHT, size=(NUM_OFFSET, 1), dtype=torch.int, device="cuda")
            offsets = torch.cat([offsets_w, offsets_h], dim=1)
        else:
            offsets_w = (torch.rand(NUM_OFFSET, 1, device="cuda") - 0.5) * 2 * 1.2 * WIDTH
            offsets_h = (torch.rand(NUM_OFFSET, 1, device="cuda") - 0.5) * 2 * 1.2 * HEIGHT
            offsets = torch.cat([offsets_w, offsets_h], dim=1)
            offsets_roundint = offsets.round().int()
            for ichan in range(offsets.size(0)):
                off_x_int = offsets_roundint[ichan, 0]
                off_y_int = offsets_roundint[ichan, 1]
                if off_x_int != 0:
                    img[:, ichan, :, slice(0, -off_x_int) if off_x_int < 0 else slice(-off_x_int, None)] = 0
                    label[:, ichan, :, slice(0, -off_x_int) if off_x_int < 0 else slice(-off_x_int, None)] = 0

                if off_y_int != 0:
                    img[:, ichan, slice(0, -off_y_int) if off_y_int < 0 else slice(-off_y_int, None)] = 0
                    label[:, ichan, slice(0, -off_y_int) if off_y_int < 0 else slice(-off_y_int, None)] = 0

            offsets.requires_grad_()
        img.requires_grad_()

        if args.print:
            print("Input:")
            print(img)
            print(offsets)

        first_output = None
        first_grad = None
        if args.frac:
            first_grad_off = None

        for md in models:
            output = md(img, offsets, CHAN_PER_OFFSET)
            if args.sync:
                torch.cuda.synchronize()

            assert output.size() == label.size()
            zero_grad(img)
            if args.frac:
                zero_grad(offsets)
            output.backward(gradient=label)
            if args.sync:
                torch.cuda.synchronize()

            if args.model == "all":
                grad = img.grad.detach().cpu()
                if args.frac:
                    grad_off = offsets.grad.detach().cpu()
                output = output.detach().cpu()

                # for isamp in range(1):
                #     plt.figure()
                #     plt.imshow(img[isamp, 0].data.cpu())
                #     plt.figure()
                #     plt.imshow(normal_grad[isamp, 0].data.cpu())
                #     plt.figure()
                #     plt.imshow(cuda_grad[isamp, 0].data.cpu())
                #     fig, axes = plt.subplots(n_corr_h, n_corr_w, squeeze=False)
                #     for iy in range(n_corr_h):
                #         for ix in range(n_corr_w):
                #             ax = axes[iy, ix]
                #             ax.axis("off")
                #             ax.imshow(normal_output[isamp, 0, iy, ix].data.cpu())
                #     fig, axes = plt.subplots(n_corr_h, n_corr_w, squeeze=False)
                #     for iy in range(n_corr_h):
                #         for ix in range(n_corr_w):
                #             ax = axes[iy, ix]
                #             ax.axis("off")
                #             ax.imshow(cuda_output[isamp, 0, iy, ix].data.cpu())

                if first_output is None:
                    first_output = output
                    first_grad = grad
                    if args.frac:
                        first_grad_off = grad_off
                else:
                    # N.B. empty_like used in Displace Function could cause inaccurate here
                    # functions could reuse previous result
                    max_error_out = (output - first_output).abs().max()
                    max_error_grad = (grad - first_grad).abs().max()
                    if args.frac:
                        max_error_grad_off = (grad_off - first_grad_off).abs().max()
                    if args.relative_error:
                        max_error_out /= max(output.abs().max(), first_output.abs().max())
                        max_error_grad /= max(grad.abs().max(), first_grad.abs().max())
                        if args.frac:
                            max_error_grad_off /= max(grad_off.abs().max(), first_grad_off.abs().max())

                    if max_error_out > smax_output_error:
                        smax_output_error = max_error_out

                    if max_error_grad > smax_grad_error:
                        smax_grad_error = max_error_grad

                    if args.frac and max_error_grad_off > smax_grad_off_error:
                        smax_grad_off_error = max_error_grad_off

                    if max_error_out > 0:
                        print("Max Output Error: %e" % (max_error_out))
                        if args.print:
                            print(first_output)
                            print(output)
                        if args.raise_ and max_error_out > threshold:
                            raise ValueError("Max Output Error: %e" % (max_error_out))

                    if max_error_grad > 0:
                        print("Max Grad Error: %e" % (max_error_grad))
                        if args.print:
                            print(first_grad)
                            print(grad)
                        if args.raise_ and max_error_grad > threshold:
                            raise ValueError("Max Grad Error: %e" % (max_error_grad))

                    if args.frac:
                        if max_error_grad_off > 0:
                            print("Max Offset Grad Error: %e" % (max_error_grad_off))
                            if args.print:
                                print(first_grad_off)
                                print(grad_off)
                            if args.raise_ and max_error_grad_off > threshold:
                                raise ValueError("Max Offset Grad Error: %e" % (max_error_grad_off))
        
        # plt.show()
        print("%d test pass" % (i+1,))

    if args.time:
        end_time = datetime.datetime.now()
        avg_time = float((end_time - start_time).total_seconds()) / REPEAT_TIMES
        print("Average time: %.4f" % (avg_time,))

    if args.memory:
        import subprocess
        import os
        mem_usage_str = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        mem_usage_str = mem_usage_str.decode()
        mem_total = 0
        for l in mem_usage_str.split("\n"):
            if l.strip():
                pid, mem = l.split(", ")
                if int(pid) == os.getpid():
                    mem_total += int(mem)
        print("Mem usage: %d MB" % (mem_total,))

    print("Overall Max Output Error: %e" % (smax_output_error))
    print("Overall Max Grad Error: %e" % (smax_grad_error))
    if args.frac:
        print("Overall Max Offset Grad Error: %e" % (smax_grad_off_error))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", choices=["normal", "cuda", "all"])
    argparser.add_argument("--raise", dest="raise_", action="store_true")
    argparser.add_argument("--double", action="store_true")
    argparser.add_argument("--thr", type=str, default="0")
    argparser.add_argument("--time", action="store_true")
    argparser.add_argument("--memory", action="store_true")
    argparser.add_argument("--sync", action="store_true")
    argparser.add_argument("--seed", type=int)
    argparser.add_argument("--relative-error", action="store_true")
    argparser.add_argument("--img-value", type=float)
    argparser.add_argument("--label-value", type=float)
    argparser.add_argument("--frac", action="store_true")
    argparser.add_argument("--print", action="store_true")
    argparser.add_argument("--repeat", type=int, default=100)
    argparser.add_argument("--batch-size", type=int, default=20)
    argparser.add_argument("--num-offset", type=int, default=32)
    argparser.add_argument("--chan-per-offset", type=int, default=1)
    argparser.add_argument("--height", type=int, default=127)
    argparser.add_argument("--width", type=int, default=129)
    main(argparser.parse_args())
