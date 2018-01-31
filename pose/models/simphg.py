# code ported from pose-ae-train
from torch import nn
from torch.autograd import Function
import sys 
import time
import numpy as np
import torch

Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Full(nn.Module):
    def __init__(self, inp_dim, out_dim, bn = False, relu = False):
        super(Full, self).__init__()
        self.fc = nn.Linear(inp_dim, out_dim, bias = True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.fc(x.view(x.size()[0], -1))
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class SimpHourglass(nn.Module):
    # n: stacks number
    # f: dimension of high res. part
    # increase: increasement of dimension of low res. part
    def __init__(self, n, f, bn=None, increase=128):
        super(SimpHourglass, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = SimpHourglass(n-1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)
        self.up2  = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2