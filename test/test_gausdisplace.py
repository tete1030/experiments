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
import time

argparser = argparse.ArgumentParser()
argparser.add_argument("-b", "--batch", type=int, default=1)
argparser.add_argument("-c", "--channel", type=int, default=1)
argparser.add_argument("-H", "--height", type=int, default=30)
argparser.add_argument("-w", "--width", type=int, default=30)
argparser.add_argument("-s", "--sample", type=int, default=128)
argparser.add_argument("--pos-rand-range", type=str, default=None)
argparser.add_argument("--offset-init-rand", type=float, default=10)
argparser.add_argument("--offset-act-init", type=str, default=None)
argparser.add_argument("--offset-est-init", type=str, default=None)
argparser.add_argument("--offset-lr", type=float, default=10)
argparser.add_argument("--sigma-lr", type=float, default=20)
argparser.add_argument("--sigma-angle", type=float, default=30.)
argparser.add_argument("--sigma-scale", type=float, default=3.)
argparser.add_argument("--no-target", action="store_true")
argparser.add_argument("--point-ksize", type=int, default=3)
argparser.add_argument("--point-sigma", type=float, default=1.)
argparser.add_argument("--loss-sigma-cof", type=float, default=0)
argparser.add_argument("--no-transform-sigma", action="store_true")
argparser.add_argument("--sigma-decay-steps", type=int, default=0)
argparser.add_argument("--sim-max-pool", action="store_true")
argparser.add_argument("--sampler", choices=["gaussian", "uniform"], default="uniform")
argparser.add_argument("--weight-dist", choices=["gaussian", "uniform"], default="gaussian")
argparser.add_argument("--iter", type=int)
argparser.add_argument("--vis", action="store_true")
argparser.add_argument("--no-print", action="store_true")
args = argparser.parse_args()

BATCH_SIZE = args.batch
NUM_CHANNEL = args.channel
HEIGHT = args.height
WIDTH = args.width

NUM_SAMPLE = args.sample
POS_RAND_RANGE = json.loads(args.pos_rand_range) if args.pos_rand_range is not None else None
OFFSET_INIT_RAND = args.offset_init_rand
OFFSET_LR = args.offset_lr
SIGMA_LR = args.sigma_lr
SIGMA_ANGLE = args.sigma_angle / 180 * np.pi
SIGMA_SCALE = args.sigma_scale
POINT_KSIZE = args.point_ksize
POINT_SIGMA = args.point_sigma

assert not (args.loss_sigma_cof > 0 and args.sigma_decay_steps > 0)

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

displacer = PositionalGaussianDisplaceModule(NUM_CHANNEL, NUM_SAMPLE, SIGMA_ANGLE, SIGMA_SCALE, sampler=args.sampler, weight_dist=args.weight_dist, learnable_sigma=not bool(args.sigma_decay_steps > 0), transform_sigma=not args.no_transform_sigma, fill=1 if args.sim_max_pool else 0).cuda()
blurer = GaussianBlur(NUM_CHANNEL, POINT_KSIZE, POINT_SIGMA).cuda()
optimizer = SGD([
    {"params": list(displacer.parameters()), "lr": SIGMA_LR},
    {"params": [estimated_offsets], "lr": OFFSET_LR},
])

sigint_triggered = 0
ori_sigint_handler = signal.getsignal(signal.SIGINT)
def on_sigint(sig, frame):
    global sigint_triggered
    sigint_triggered += 1
    if sigint_triggered == 3:
        ori_sigint_handler(sig, frame)
signal.signal(signal.SIGINT, on_sigint)

do_break = False
i = 0
start_time = time.time()
while True:
    if i == 1:
        start_time = time.time()
    if do_break or (args.iter is not None and i >= args.iter):
        break
    if i % 10000 == 0:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] / 2
        optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] / 2
    inp = torch.zeros((BATCH_SIZE, NUM_CHANNEL, HEIGHT, WIDTH), dtype=torch.float, device="cuda")
    targ = torch.zeros((BATCH_SIZE, NUM_CHANNEL, HEIGHT, WIDTH), dtype=torch.float, device="cuda")

    batch_ind = torch.arange(BATCH_SIZE, dtype=torch.long, device="cuda")[:, None].repeat(1, NUM_CHANNEL).view(-1)
    channel_ind = torch.arange(NUM_CHANNEL, dtype=torch.long, device="cuda")[None, :].repeat(BATCH_SIZE, 1).view(-1)
    if POS_RAND_RANGE is None:
        x = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * WIDTH).long().view(-1)
        y = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * HEIGHT).long().view(-1)
    else:
        x = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * (POS_RAND_RANGE[0][1] - POS_RAND_RANGE[0][0]) + POS_RAND_RANGE[0][0]).long().view(-1)
        y = (torch.rand((BATCH_SIZE, NUM_CHANNEL), dtype=torch.float, device="cuda") * (POS_RAND_RANGE[1][1] - POS_RAND_RANGE[1][0]) + POS_RAND_RANGE[1][0]).long().view(-1)
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
            if sigint_triggered > 0 and args.vis:
                retained_grad = True
                estimated_offsets_x.retain_grad()
                estimated_offsets_y.retain_grad()
            if args.sim_max_pool:
                inp = inp.exp()
            out = displacer(inp, estimated_offsets_x, estimated_offsets_y, 1)
            if args.sim_max_pool:
                out = out.log()
            assert not torch.isnan(out).any(), "out being nan: {}/{}".format(torch.isnan(out).long().sum(), out.numel())
            loss = (out - targ).pow(2).mean(dim=-1).mean(dim=-1).mean(dim=0).sum()
            if args.loss_sigma_cof > 0:
                loss = loss + (displacer.angle_std().pow(2).sum() + displacer.scale_std().pow(2).sum()) * args.loss_sigma_cof
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.sigma_decay_steps > 0:
                displacer.set_angle_std(SIGMA_ANGLE * (float(args.sigma_decay_steps - i - 1) / args.sigma_decay_steps))
                displacer.set_scale_std(SIGMA_SCALE * (float(args.sigma_decay_steps - i - 1) / args.sigma_decay_steps))

    if not args.no_print:
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

    if sigint_triggered > 0:
        if args.vis and retained_grad:
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
        else:
            do_break = True

    i += 1

end_time = time.time()
print("Time: {}s".format(end_time - start_time))
