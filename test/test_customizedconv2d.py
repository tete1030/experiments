#!python3
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from lib.models.displace import CustomizedGradConv2dCUDNN, CustomizedGradDepthwiseConv2d
import argparse
import datetime

def zero_grad(t):
    if t.grad is not None:
        t.grad.zero_()

class Model(nn.Module):
    def __init__(self, fn):
        super(Model, self).__init__()
        self.fn = fn

    def forward(self, *args):
        return self.fn(*args)

def main(args):
    REPEAT_TIMES = 100
    
    BATCH_SIZE = 1
    CHAN_PER_OFFSET = 3
    NUM_OFFSET = 1
    CHANNEL_SIZE = CHAN_PER_OFFSET * NUM_OFFSET
    WIDTH = 124
    HEIGHT = 129

    KERNEL_WIDTH = 5
    KERNEL_HEIGHT = 7
    DILATION_WIDTH = 5
    DILATION_HEIGHT = 3
    STRIDE_WIDTH = 1
    STRIDE_HEIGHT = 1

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
        functions.append(F.conv2d)
    elif args.model == "cuda":
        functions.append(CustomizedGradDepthwiseConv2d.apply)
    elif args.model == "cudnn":
        functions.append(CustomizedGradConv2dCUDNN.apply)
    elif args.model == "nocudnn":
        functions.extend([F.conv2d, CustomizedGradDepthwiseConv2d.apply])
    elif args.model == "nonormal":
        functions.extend([CustomizedGradDepthwiseConv2d.apply, CustomizedGradConv2dCUDNN.apply])
    elif args.model == "all":
        functions.extend([F.conv2d, CustomizedGradDepthwiseConv2d.apply, CustomizedGradConv2dCUDNN.apply])

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

    if args.time:
        start_time = datetime.datetime.now()

    for i in range(REPEAT_TIMES):
        img = torch.ones(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        label = torch.ones(BATCH_SIZE, CHANNEL_SIZE, (HEIGHT + STRIDE_HEIGHT - 1) // STRIDE_HEIGHT, (WIDTH + STRIDE_WIDTH - 1) // STRIDE_WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        kernel = torch.ones(CHANNEL_SIZE, 1, KERNEL_HEIGHT, KERNEL_WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        offsets_w = torch.randint(low=-WIDTH, high=WIDTH, size=(NUM_OFFSET, 1), dtype=torch.int, device="cuda")
        offsets_h = torch.randint(low=-HEIGHT, high=HEIGHT, size=(NUM_OFFSET, 1), dtype=torch.int, device="cuda")
        offsets = torch.cat([offsets_w, offsets_h], dim=1)

        # img = torch.ones(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        # label = torch.ones(BATCH_SIZE, n_corr_h, n_corr_w, CHANNEL_SIZE, KERNEL_HEIGHT, KERNEL_WIDTH, dtype=DTYPE, device="cuda")
        
        # img = torch.zeros(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        # for isamp in range(BATCH_SIZE):
        #     for ichan in range(CHANNEL_SIZE):
        #         randX = torch.randint(WIDTH, size=(100,), dtype=torch.long)
        #         randY = torch.randint(HEIGHT, size=(100,), dtype=torch.long)
        #         img.data[isamp, ichan, randY, randX] = 1
        # label = torch.zeros(BATCH_SIZE, n_corr_h, n_corr_w, CHANNEL_SIZE, KERNEL_HEIGHT, KERNEL_WIDTH, dtype=DTYPE, device="cuda")
        
        first_output = None
        first_grad = None

        for md in models:
            extra_back = False
            if args.mask:
                if md.fn is not F.conv2d:
                    output = md(img, kernel, None, (STRIDE_HEIGHT, STRIDE_WIDTH), (KERNEL_HEIGHT // 2 * DILATION_HEIGHT, KERNEL_WIDTH // 2 * DILATION_WIDTH), (DILATION_HEIGHT, DILATION_WIDTH), CHANNEL_SIZE, offsets, CHAN_PER_OFFSET, (KERNEL_HEIGHT // 2 * DILATION_HEIGHT, KERNEL_WIDTH // 2 * DILATION_WIDTH))
                else:
                    extra_back = True
                    output = md(img, kernel, None, (STRIDE_HEIGHT, STRIDE_WIDTH), (KERNEL_HEIGHT // 2 * DILATION_HEIGHT, KERNEL_WIDTH // 2 * DILATION_WIDTH), (DILATION_HEIGHT, DILATION_WIDTH), CHANNEL_SIZE)
            else:
                output = md(img, kernel, None, (STRIDE_HEIGHT, STRIDE_WIDTH), (KERNEL_HEIGHT // 2 * DILATION_HEIGHT, KERNEL_WIDTH // 2 * DILATION_WIDTH), (DILATION_HEIGHT, DILATION_WIDTH), CHANNEL_SIZE)

            if args.sync:
                torch.cuda.synchronize()

            assert output.size() == label.size()
            zero_grad(img)
            zero_grad(kernel)
            output.backward(gradient=label, retain_graph=extra_back)
            if args.sync:
                torch.cuda.synchronize()

            if extra_back:
                zero_grad(kernel)
                img.requires_grad = False
                new_label = torch.zeros_like(label)
                for ioff in range(len(offsets)):
                    off_x = offsets[ioff, 0]
                    off_y = offsets[ioff, 1]
                    min_x, max_x = max(0, off_x+KERNEL_WIDTH//2*DILATION_WIDTH), min(WIDTH, WIDTH+off_x-KERNEL_WIDTH//2*DILATION_WIDTH)
                    min_x, max_x = min(WIDTH, min_x), max(0, max_x)
                    min_y, max_y = max(0, off_y+KERNEL_HEIGHT//2*DILATION_HEIGHT), min(HEIGHT, HEIGHT+off_y-KERNEL_HEIGHT//2*DILATION_HEIGHT)
                    min_y, max_y = min(HEIGHT, min_y), max(0, max_y)
                    new_label[:, ioff*CHAN_PER_OFFSET:ioff*CHAN_PER_OFFSET+CHAN_PER_OFFSET, min_y:max_y, min_x:max_x] = \
                        label[:, ioff*CHAN_PER_OFFSET:ioff*CHAN_PER_OFFSET+CHAN_PER_OFFSET, min_y:max_y, min_x:max_x]
                output.backward(gradient=new_label)
                img.requires_grad = True

            if args.model in ["all", "nocudnn", "nonormal"]:
                grad_img = img.grad.detach().cpu()
                grad_kernel = kernel.grad.detach().cpu()
                output = output.detach().cpu()

                # import ipdb; ipdb.set_trace()

                if first_output is None:
                    first_output = output
                    first_grad_img = grad_img
                    first_grad_kernel = grad_kernel
                else:
                    # N.B. empty_like used in Displace Function could cause inaccurate here
                    # functions could reuse previous result
                    max_error_out = (output - first_output).abs().max()
                    max_error_grad_img = (grad_img - first_grad_img).abs().max()
                    max_error_grad_kernel = (grad_kernel - first_grad_kernel).abs().max()
                    if args.raise_ and max_error_out > threshold:
                        raise ValueError("Max Output Error: %e" % (max_error_out))
                    elif max_error_out > 0:
                        print("Max Output Error: %e" % (max_error_out))

                    if args.raise_ and max_error_grad_img > threshold:
                        raise ValueError("Max Img Grad Error: %e" % (max_error_grad_img))
                    elif max_error_grad_img > 0:
                        print("Max Img Grad Error: %e" % (max_error_grad_img))

                    if args.raise_ and max_error_grad_kernel > threshold:
                        raise ValueError("Max Kernel Grad Error: %e" % (max_error_grad_kernel))
                    elif max_error_grad_kernel > 0:
                        print("Max Kernel Grad Error: %e" % (max_error_grad_kernel))
        
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", choices=["normal", "cuda", "all", "nocudnn", "nonormal"])
    argparser.add_argument("--raise", dest="raise_", action="store_true")
    argparser.add_argument("--double", action="store_true")
    argparser.add_argument("--thr", choices=["0", "eps32", "eps64"], default="0")
    argparser.add_argument("--time", action="store_true")
    argparser.add_argument("--memory", action="store_true")
    argparser.add_argument("--sync", action="store_true")
    argparser.add_argument("--mask", action="store_true")
    argparser.add_argument("--seed", type=int)
    main(argparser.parse_args())
