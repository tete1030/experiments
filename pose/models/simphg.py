# code ported from pose-ae-train
from torch import nn
from torch.autograd import Function
import sys 
import time
import numpy as np
import torch

__all__ = ["Conv", "Pool", "SimpHourglass"]

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
    def __init__(self, inp_dim, n, bn=None, increase=128):
        """Hourglass with out residual module
        
        Arguments:
            inp_dim {int} -- dimension of high res. part
            n {int} -- scale number
        
        Keyword Arguments:
            bn {bool} -- batch normalization module (default: {None})
            increase {int} -- increasement of dimension of low res. part (default: {128})
        """

        super(SimpHourglass, self).__init__()
        low_dim = inp_dim + increase
        self.up1 = Conv(inp_dim, inp_dim, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(inp_dim, low_dim, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = SimpHourglass(low_dim, n-1, bn=bn)
        else:
            self.low2 = Conv(low_dim, low_dim, 3, bn=bn)
        self.low3 = Conv(low_dim, inp_dim, 3)
        # TODO: COMPATIBILITY Change to new module
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2