import torch
from torch import nn
import numpy as np

class SpaceNormalization(nn.Module):
    def __init__(self):
        super(SpaceNormalization, self).__init__()

    def forward(self, x):
        x = x + torch.tensor(np.finfo(np.float32).eps, device=x.device, dtype=torch.float)
        return x / x.sum(-1, keepdim=True).sum(-2, keepdim=True)
