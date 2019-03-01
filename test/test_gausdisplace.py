import numpy as np
import torch
from torch import nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import sys, os
import argparse
import signal
import json
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")))
from lib.models.displacechan import PositionalGaussianDisplaceModule
from lib.models.gaussianblur import GaussianBlur

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--batch", type=int, default=1)
argparser.add_argument("-c", "--channel", type=int, default=1)
argparser.add_argument("-H", "--height", type=int, default=30)
argparser.add_argument("-w", "--width", type=int, default=30)
argparser.add_argument("-s", "--sample", type=int, default=128)
argparser.add_argument("--offset-init-rand", type=float, default=10)
argparser.add_argument("--offset-act-init", type=str, default=None)
argparser.add_argument("--offset-est-init", type=str, default=None)
argparser.add_argument("--offset-lr", type=float, default=10)
argparser.add_argument("--sigma-lr", type=float, default=20)
argparser.add_argument("--sigma-angle", type=float, default=30.)
argparser.add_argument("--sigma-scale", type=float, default=3.)
argparser.add_argument("--no-target", action="store_true")
args = argparser.parse_args()

BATCH_SIZE = args.batch
NUM_CHANNEL = args.channel
HEIGHT = args.height
WIDTH = args.width

NUM_SAMPLE = args.sample
OFFSET_INIT_RAND = args.offset_init_rand
OFFSET_LR = args.offset_lr
SIGMA_LR = args.sigma_lr
SIGMA_ANGLE = args.sigma_angle
SIGMA_SCALE = args.sigma_scale

if args.offset_act_init is None:
    actual_offsets = (torch.rand((NUM_CHANNEL, 2), dtype=torch.float, device="cuda") * OFFSET_INIT_RAND - OFFSET_INIT_RAND / 2).long()
else:
    actual_offsets_val = json.loads(args.offset_act_init)
    actual_offsets = torch.tensor(actual_offsets_val, dtype=torch.long, device="cuda")

if args.offset_est_init is None:
    estimated_offsets = nn.Parameter(torch.rand((NUM_CHANNEL, 2), dtype=torch.float, device="cuda") * OFFSET_INIT_RAND - OFFSET_INIT_RAND / 2)
else:
    estimated_offsets_val = json.loads(args.offset_est_init)
    estimated_offsets = nn.Parameter(torch.tensor(estimated_offsets_val, dtype=torch.float, device="cuda"))

displacer = PositionalGaussianDisplaceModule(NUM_CHANNEL, NUM_SAMPLE, SIGMA_ANGLE / 180 * np.pi, SIGMA_SCALE, fill=0).cuda()
blurer = GaussianBlur(NUM_CHANNEL, 3, 1).cuda()
optimizer = SGD([
    {"params": list(displacer.parameters()), "lr": SIGMA_LR},
    {"params": [estimated_offsets], "lr": OFFSET_LR},
])

show_gradmap = False
ori_sigint_handler = signal.getsignal(signal.SIGINT)
def on_sigint(sig, frame):
    global show_gradmap
    if show_gradmap == False:
        show_gradmap = True
    else:
        ori_sigint_handler(sig, frame)
signal.signal(signal.SIGINT, on_sigint)

i = 0
while True:
    if i % 10000 == 0:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2
        optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] / 2
    inp = torch.zeros((BATCH_SIZE, NUM_CHANNEL, HEIGHT, WIDTH), dtype=torch.float, device="cuda")
    targ = torch.zeros((BATCH_SIZE, NUM_CHANNEL, HEIGHT, WIDTH), dtype=torch.float, device="cuda")

    batch_ind = torch.arange(BATCH_SIZE, dtype=torch.long, device="cuda")[:, None].repeat(1, NUM_CHANNEL).view(-1)
    channel_ind = torch.arange(NUM_CHANNEL, dtype=torch.long, device="cuda")[None, :].repeat(BATCH_SIZE, 1).view(-1)
    x = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * WIDTH).long().view(-1)
    y = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * HEIGHT).long().view(-1)
    x_out = (x + actual_offsets[..., 0].repeat(BATCH_SIZE).view(-1))
    y_out = (y + actual_offsets[..., 1].repeat(BATCH_SIZE).view(-1))
    mask_inp = ((x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT))
    mask_out = ((x_out >= 0) & (x_out < WIDTH) & (y_out >= 0) & (y_out < HEIGHT))
    mask = (mask_inp & mask_out)

    inp[batch_ind[mask], channel_ind[mask], y[mask], x[mask]] = 1
    inp = blurer(inp)

    if not args.no_target:
        targ[batch_ind[mask], channel_ind[mask], y_out[mask], x_out[mask]] = 1
        targ = blurer(targ)

    retained_grad = False
    with torch.autograd.enable_grad():
        with torch.autograd.detect_anomaly():
            estimated_offsets_x = estimated_offsets[None, :, None, None, 0].repeat(BATCH_SIZE, 1, HEIGHT, WIDTH)
            estimated_offsets_y = estimated_offsets[None, :, None, None, 1].repeat(BATCH_SIZE, 1, HEIGHT, WIDTH)
            if show_gradmap:
                retained_grad = True
                estimated_offsets_x.retain_grad()
                estimated_offsets_y.retain_grad()
            out = displacer(inp, estimated_offsets_x, estimated_offsets_y, 1)
            loss = (out - targ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=0).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if show_gradmap and retained_grad:
        plt.figure()
        plt.imshow(inp[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.figure()
        plt.imshow(targ[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.figure()
        plt.imshow(out[0, 0].detach().cpu(), vmin=0, vmax=1)
        plt.figure()
        X, Y = np.meshgrid(np.arange(estimated_offsets_x.size(-1)), np.arange(estimated_offsets_x.size(-2)))
        plt.quiver(X, Y, estimated_offsets_x.grad.detach().cpu().numpy()[0, 0], estimated_offsets_y.grad.detach().cpu().numpy()[0, 0])
        plt.show()

    # plt.figure()
    # plt.imshow(inp[0, 0].detach().cpu(), vmin=0, vmax=1)
    # plt.figure()
    # plt.imshow(targ[0, 0].detach().cpu(), vmin=0, vmax=1)
    # plt.figure()
    # plt.imshow(out[0, 0].detach().cpu(), vmin=0, vmax=1)
    # plt.show()

    offsets_diff = (estimated_offsets - actual_offsets.float()).pow(2).mean().item()

    print("{iter:6}: loss={loss: >2.8f}, offsets_diff={offsets_diff: >4.5f}, offset=({offset_x: > 2.3f}, {offset_y: > 2.3f}), act_offset=({act_offset_x:2}, {act_offset_y:2}), ang_sig={angle_sig: > 2.3f}, sca_sig={scale_sig: >2.3f}".format(
        iter=i,
        loss=loss.item(),
        offsets_diff=offsets_diff,
        offset_x=estimated_offsets[0, 0].item(),
        offset_y=estimated_offsets[0, 1].item(),
        act_offset_x=actual_offsets[0, 0].item(),
        act_offset_y=actual_offsets[0, 1].item(),
        angle_sig=displacer.angle_std()[0].item(),
        scale_sig=displacer.scale_std()[0].item()))
    i += 1
