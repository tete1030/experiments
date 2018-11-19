#!python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from lib.models.lacorr2d import LocalAutoCorr2DCUDA, LocalAutoCorr2D, PadInfo
import argparse
import datetime

def zero_grad(t):
    if t.grad is not None:
        t.grad.detach_()
        t.grad.zero_()

def main(args):
    REPEAT_TIMES = 100
    
    BATCH_SIZE = 8
    CHANNEL_SIZE = 16
    WIDTH = 64
    HEIGHT = 64
    KERNEL_WIDTH = 8
    KERNEL_HEIGHT = 8
    STRIDE_WIDTH = 4
    STRIDE_HEIGHT = 4
    # BATCH_SIZE = 8
    # CHANNEL_SIZE = 16
    # WIDTH = 55
    # HEIGHT = 89
    # KERNEL_WIDTH = 10
    # KERNEL_HEIGHT = 17
    # STRIDE_WIDTH = 3
    # STRIDE_HEIGHT = 5

    PAD_TOP = 4
    PAD_BOTTOM = 3
    PAD_LEFT = 4
    PAD_RIGHT = 3

    n_corr_w = int(WIDTH + PAD_LEFT + PAD_RIGHT - KERNEL_WIDTH) // int(STRIDE_WIDTH) + 1;
    n_corr_h = int(HEIGHT + PAD_TOP + PAD_BOTTOM - KERNEL_HEIGHT) // int(STRIDE_HEIGHT) + 1;

    model_constructors = list()
    models = list()

    assert not (args.model == "all" and (args.memory or args.time))

    if args.model == "normal":
        model_constructors.append(LocalAutoCorr2D)
    elif args.model == "cuda":
        model_constructors.append(LocalAutoCorr2DCUDA)
    elif args.model == "all":
        model_constructors.extend([LocalAutoCorr2D, LocalAutoCorr2DCUDA])

    for mc in model_constructors:
        models.append(mc((KERNEL_HEIGHT, KERNEL_WIDTH), (STRIDE_HEIGHT, STRIDE_WIDTH), PadInfo(PAD_TOP, PAD_BOTTOM, PAD_LEFT, PAD_RIGHT)).cuda())

    if args.time:
        start_time = datetime.datetime.now()

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

    for i in range(REPEAT_TIMES):
        img = torch.rand(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda", requires_grad=True)
        img.data += 1
        label = torch.rand(BATCH_SIZE, n_corr_h, n_corr_w, CHANNEL_SIZE, KERNEL_HEIGHT, KERNEL_WIDTH, dtype=DTYPE, device="cuda")
        label.data += 1

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
            output = md(img)

            assert output.size() == label.size()
            loss = (output - label).pow(2).mean().sqrt()
            zero_grad(img)
            # output.backward(gradient=label)
            loss.backward()

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
    main(argparser.parse_args())
