#!python3
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from lib.models.displace import Displace, DisplaceCUDA
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
        return self.fn.apply(*args)

def main(args):
    REPEAT_TIMES = 100
    
    BATCH_SIZE = 16
    CHAN_PER_OFFSET = 3
    NUM_OFFSET = 100
    CHANNEL_SIZE = CHAN_PER_OFFSET * NUM_OFFSET
    WIDTH = 128
    HEIGHT = 257
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
        functions.append(Displace)
    elif args.model == "cuda":
        functions.append(DisplaceCUDA)
    elif args.model == "all":
        functions.extend([Displace, DisplaceCUDA])

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
        img = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        label = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
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
            output = md(img, offsets, CHAN_PER_OFFSET)
            if args.sync:
                torch.cuda.synchronize()

            assert output.size() == label.size()
            zero_grad(img)
            output.backward(gradient=label)
            if args.sync:
                torch.cuda.synchronize()

            if args.model == "all":
                grad = img.grad.detach().cpu()
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
                else:
                    # N.B. empty_like used in Displace Function could cause inaccurate here
                    # functions could reuse previous result
                    max_error_out = (output - first_output).abs().max()
                    max_error_grad = (grad - first_grad).abs().max()
                    if args.raise_ and max_error_out > threshold:
                        raise ValueError("Max Output Error: %e" % (max_error_out))
                    elif max_error_out > 0:
                        print("Max Output Error: %e" % (max_error_out))

                    if args.raise_ and max_error_grad > threshold:
                        raise ValueError("Max Grad Error: %e" % (max_error_grad))
                    elif max_error_grad > 0:
                        print("Max Grad Error: %e" % (max_error_grad))
        
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
    argparser.add_argument("--model", choices=["normal", "cuda", "all"])
    argparser.add_argument("--raise", dest="raise_", action="store_true")
    argparser.add_argument("--double", action="store_true")
    argparser.add_argument("--thr", choices=["0", "eps32", "eps64"], default="0")
    argparser.add_argument("--time", action="store_true")
    argparser.add_argument("--memory", action="store_true")
    argparser.add_argument("--sync", action="store_true")
    argparser.add_argument("--seed", type=int)
    main(argparser.parse_args())
