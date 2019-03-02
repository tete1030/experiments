import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair

class GaussianBlur(nn.Module):
    def __init__(self, inplanes=None, kernel_size=3, sigma=1):
        super(GaussianBlur, self).__init__()
        kernel_size = _pair(kernel_size)
        self.inplanes = inplanes
        self.kernel_size = kernel_size
        if kernel_size == (1, 1):
            self.gaussian_kernel = None
        else:
            assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
            y = torch.arange(-int(kernel_size[0] // 2), int(kernel_size[0] // 2) + 1)
            x = torch.arange(-int(kernel_size[1] // 2), int(kernel_size[1] // 2) + 1)
            field = torch.stack([x.expand(kernel_size[0], -1), y[:, None].expand(-1, kernel_size[1])], dim=2).float()
            if inplanes is None:
                inplanes = 1
            self.register_buffer("gaussian_kernel", torch.exp(- field.pow(2).sum(dim=2) / 2 / float(sigma) ** 2).view(1, 1, kernel_size[0], kernel_size[1]).repeat(inplanes, 1, 1, 1))

    def forward(self, x):
        if self.kernel_size == (1, 1):
            return x
        inplanes = x.size(1)
        if inplanes == self.inplanes:
            gaussian_kernel = self.gaussian_kernel
        else:
            gaussian_kernel = self.gaussian_kernel[:1].repeat(inplanes, 1, 1, 1)
        return F.conv2d(x, gaussian_kernel, padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2), groups=inplanes)
