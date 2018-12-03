import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

class GaussianBlur(nn.Module):
    def __init__(self, inplanes, kernel_size=3, sigma=1):
        super(GaussianBlur, self).__init__()
        kernel_size = _pair(kernel_size)
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        y = torch.arange(-int(kernel_size[0] // 2), int(kernel_size[0] // 2) + 1)
        x = torch.arange(-int(kernel_size[1] // 2), int(kernel_size[1] // 2) + 1)
        field = torch.stack([x.expand(kernel_size[0], -1), y[:, None].expand(-1, kernel_size[1])], dim=2).float()
        self.inplanes = inplanes
        self.kernel_size = kernel_size
        self.register_buffer("gaussian_kernel", torch.exp(- field.pow(2).sum(dim=2) / 2 / float(sigma) ** 2).view(1, 1, kernel_size[0], kernel_size[1]).repeat(self.inplanes, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.gaussian_kernel, padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2), groups=self.inplanes)
