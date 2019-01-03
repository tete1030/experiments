#!python3
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import datetime
from torch.optim import SGD, Adam
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from lib.models.displace import DisplaceFracCUDA, PositionalDisplace
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

def compare(args):
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
        functions.append(DisplaceFracCUDA)
    elif args.model == "pos":
        functions.append(PositionalDisplace)
    elif args.model == "all":
        functions.extend([DisplaceFracCUDA, PositionalDisplace])

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
    smax_grad_off_error = 0.

    for i in range(REPEAT_TIMES):
        img = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda")
        if args.img_value is not None:
            img.fill_(args.img_value)
        label = torch.randn(BATCH_SIZE, CHANNEL_SIZE, HEIGHT, WIDTH, dtype=DTYPE, device="cuda")
        if args.label_value is not None:
            label.fill_(args.label_value)


        offsets_w = (torch.rand(NUM_OFFSET, 1, device="cuda") - 0.5) * 2 * 1.2 * WIDTH
        offsets_h = (torch.rand(NUM_OFFSET, 1, device="cuda") - 0.5) * 2 * 1.2 * HEIGHT
        offsets = torch.cat([offsets_w, offsets_h], dim=1)
        offsets.requires_grad_()
        img.requires_grad_()

        if args.print:
            print("Input:")
            print(img)
            print(offsets)

        first_output = None
        first_grad = None
        first_grad_off = None

        for md in models:
            output = md(img, offsets, CHAN_PER_OFFSET)
            if args.sync:
                torch.cuda.synchronize()

            assert output.size() == label.size()
            zero_grad(img)
            zero_grad(offsets)
            output.backward(gradient=label)
            if args.sync:
                torch.cuda.synchronize()

            if args.model == "all":
                grad = img.grad.detach().cpu()
                grad_off = offsets.grad.detach().cpu()
                output = output.detach().cpu()

                if first_output is None:
                    first_output = output
                    first_grad = grad
                    first_grad_off = grad_off
                else:
                    # N.B. empty_like used in Displace Function could cause inaccurate here
                    # functions could reuse previous result
                    max_error_out = (output - first_output).abs().max()
                    max_error_grad = (grad - first_grad).abs().max()
                    max_error_grad_off = (grad_off - first_grad_off).abs().max()
                    if args.relative_error:
                        max_error_out /= max(output.abs().max(), first_output.abs().max())
                        max_error_grad /= max(grad.abs().max(), first_grad.abs().max())
                        max_error_grad_off /= max(grad_off.abs().max(), first_grad_off.abs().max())

                    if max_error_out > smax_output_error:
                        smax_output_error = max_error_out

                    if max_error_grad > smax_grad_error:
                        smax_grad_error = max_error_grad

                    if max_error_grad_off > smax_grad_off_error:
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
    print("Overall Max Offset Grad Error: %e" % (smax_grad_off_error))

def make_gaussian_kernel(inplanes, kernel_size, sigma):
    from torch.nn.modules.utils import _pair
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    y = torch.arange(-int(kernel_size[0] // 2), int(kernel_size[0] // 2) + 1)
    x = torch.arange(-int(kernel_size[1] // 2), int(kernel_size[1] // 2) + 1)
    field = torch.stack([x.expand(kernel_size[0], -1), y[:, None].expand(-1, kernel_size[1])], dim=2).float()
    kernel = torch.exp(- field.pow(2).sum(dim=2) / 2 / float(sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size[0], kernel_size[1]).repeat(inplanes, 1, 1, 1)

def make_cone_kernel(inplanes, kernel_size, sigma):
    from torch.nn.modules.utils import _pair
    kernel_size = _pair(kernel_size)
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    y = torch.arange(-int(kernel_size[0] // 2), int(kernel_size[0] // 2) + 1, dtype=torch.float) / float(sigma)
    x = torch.arange(-int(kernel_size[1] // 2), int(kernel_size[1] // 2) + 1, dtype=torch.float) / float(sigma)
    field = torch.stack([x.expand(kernel_size[0], -1), y[:, None].expand(-1, kernel_size[1])], dim=2)
    kernel = (1 - field.pow(2).sum(dim=2).sqrt()).clamp(min=0)
    kernel /= kernel.sum()
    return kernel.view(1, 1, kernel_size[0], kernel_size[1]).repeat(inplanes, 1, 1, 1)

def show(args):
    import cv2
    import matplotlib.pyplot as plt
    from lib.models.gaussianblur import GaussianBlur

    if args.seed is None:
        torch.cuda.seed_all()
    else:
        print("Using seed {}".format(args.seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
    # initial_seed will initialize CUDA
    print("seed = {}".format(torch.cuda.initial_seed()))

    if args.image:
        img = cv2.imread("test/test.jpg")[:, :, ::-1].astype(np.float32) / 255
        img = np.moveaxis(img, -1, 0)

    if args.generate:
        WIDTH = args.width
        HEIGHT = args.height
        img = (np.stack((
            np.sqrt((np.arange(WIDTH).reshape((1, WIDTH)) / WIDTH) ** 2 + (np.arange(HEIGHT).reshape((HEIGHT, 1)) / HEIGHT) ** 2),
            np.sqrt((np.arange(WIDTH).reshape((1, WIDTH)) / WIDTH - 1) ** 2 + (np.arange(HEIGHT).reshape((HEIGHT, 1)) / HEIGHT) ** 2),
            np.sqrt((np.arange(WIDTH).reshape((1, WIDTH)) / WIDTH) ** 2 + (np.arange(HEIGHT).reshape((HEIGHT, 1)) / HEIGHT - 1) ** 2)), axis=0) / np.sqrt(2)).astype(np.float32)

    if args.vis:
        plt.imshow(np.moveaxis(img, 0, -1))
        plt.show()

    opt_image = False
    opt_offset = False
    if args.opt == "image":
        opt_image = True
    elif args.opt == "offset":
        opt_offset = True
    else:
        opt_image = True
        opt_offset = True

    FACTOR = args.factor

    img = img.reshape((1,) + img.shape)
    img = img[..., :((img.shape[-2] // FACTOR) * FACTOR), :((img.shape[-1] // FACTOR) * FACTOR)]
    ori_img = torch.tensor(img, device="cuda", requires_grad=False)
    img = ori_img.clone().requires_grad_(opt_image)
    offsets = torch.randn((1, 2, img.size(-2) // FACTOR, img.size(-1) // FACTOR), device="cuda", requires_grad=opt_offset)
    if opt_offset and args.vis:
        plt.imshow(offsets.detach()[0].pow(2).sum(dim=0).sqrt().cpu().numpy(), vmin=0, vmax=2)
        plt.show()
    params = []
    if opt_image:
        params.append({"params": [img], "lr": args.lr_image})
    if opt_offset:
        params.append({"params": [offsets], "lr": args.lr_offset})
    optimizer = SGD(params)

    if args.backward_blur_type == "gaussian":
        make_kernel = make_gaussian_kernel
    elif args.backward_blur_type == "cone":
        make_kernel = make_cone_kernel
    if args.backward_blur_size > 1:
        blur = make_kernel(3, args.backward_blur_size, args.backward_blur_sigma).cuda()
    else:
        blur = None

    decayed = False
    for i in range(args.iter):
        if False and float(i) / args.iter > 0.7 and args.backward_blur_size > 1 and not decayed:
            decayed = True
            blur = make_kernel(3, args.backward_blur_size, args.backward_blur_sigma / 2).cuda()

        up_offsets = torch.nn.functional.interpolate(offsets * args.offset_sigma, scale_factor=FACTOR, mode="bilinear")
        up_offsets = up_offsets.view(up_offsets.size(0), 2, -1, *up_offsets.size()[2:])
        up_offsets = torch.stack((up_offsets[:, 0], up_offsets[:, 1]), dim=-1)
        new_img = PositionalDisplace.apply(img, up_offsets, 3, blur)
        diff = new_img - ori_img
        loss = diff.pow(2).sum()
        if i % args.out_int == 0:
            if opt_image and args.vis:
                plt.imshow(np.moveaxis(img.detach().cpu().numpy()[0], 0, -1))
                plt.show()
            if opt_offset and args.vis:
                plt.imshow(up_offsets[0, 0].detach().pow(2).sum(dim=-1).sqrt().cpu().numpy(), vmin=0, vmax=args.offset_sigma * 2)
                plt.show()
            if opt_offset and args.log:
                print("offset_sum: " + str(offsets.detach().abs().sum().item()))
            if args.vis:
                plt.imshow(np.moveaxis(new_img.detach().cpu().numpy()[0], 0, -1))
                plt.show()
                plt.imshow(np.moveaxis(diff.detach().abs().cpu().numpy()[0], 0, -1))
                plt.show()
            if args.log:
                print("loss: " + str(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    subparsers = argparser.add_subparsers()
    compare_parser = subparsers.add_parser("comp")
    compare_parser.add_argument("--model", choices=["normal", "pos", "all"])
    compare_parser.add_argument("--raise", dest="raise_", action="store_true")
    compare_parser.add_argument("--double", action="store_true")
    compare_parser.add_argument("--thr", type=str, default="0")
    compare_parser.add_argument("--time", action="store_true")
    compare_parser.add_argument("--memory", action="store_true")
    compare_parser.add_argument("--sync", action="store_true")
    compare_parser.add_argument("--seed", type=int)
    compare_parser.add_argument("--relative-error", action="store_true")
    compare_parser.add_argument("--img-value", type=float)
    compare_parser.add_argument("--label-value", type=float)
    compare_parser.add_argument("--print", action="store_true")
    compare_parser.add_argument("--repeat", type=int, default=100)
    compare_parser.add_argument("--batch-size", type=int, default=20)
    compare_parser.add_argument("--num-offset", type=int, default=32)
    compare_parser.add_argument("--chan-per-offset", type=int, default=1)
    compare_parser.add_argument("--height", type=int, default=127)
    compare_parser.add_argument("--width", type=int, default=129)
    compare_parser.set_defaults(func=compare)

    show_parser = subparsers.add_parser("show")
    group = show_parser.add_mutually_exclusive_group()
    group.add_argument("--image", action="store_true")
    group.add_argument("--generate", action="store_true")
    show_parser.add_argument("--width", type=int, default=128)
    show_parser.add_argument("--height", type=int, default=128)
    show_parser.add_argument("--opt", choices=["offset", "image", "both"])
    show_parser.add_argument("--lr-offset", type=float, default=1e-1)
    show_parser.add_argument("--lr-image", type=float, default=1e-3)
    show_parser.add_argument("--factor", type=int)
    show_parser.add_argument("--iter", type=int)
    show_parser.add_argument("--out-int", type=int)
    show_parser.add_argument("--offset-sigma", type=float)
    show_parser.add_argument("--backward-blur-type", choices=["gaussian", "cone"])
    show_parser.add_argument("--backward-blur-size", type=int, default=1)
    show_parser.add_argument("--backward-blur-sigma", type=float, default=1)
    show_parser.add_argument("--seed", type=int)
    show_parser.add_argument("--vis", action="store_true")
    show_parser.add_argument("--log", action="store_true")
    show_parser.set_defaults(func=show)
    args = argparser.parse_args()
    args.func(args)
