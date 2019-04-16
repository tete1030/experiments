import os
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import truncnorm
from torch.utils.data import Dataset
from lib.utils.transforms import fliplr_chwimg, fliplr_pts, get_transform, transform
from torch.nn.functional import affine_grid, grid_sample

def transform_maps(x, scale, rotate, blur_sigma=None, translation_factor=None):
    batch_size = x.size(0)
    in_channels = x.size(1)
    height = x.size(2)
    width = x.size(3)
    half_height = height / 2
    half_width = width / 2

    scale_mat = torch.zeros(batch_size, 3, 3).to(x.device, non_blocking=True)

    if translation_factor is None:
        translation_factor = torch.tensor([[0, 0]], dtype=torch.float).repeat(batch_size, 1)

    # transform coordinates -> translation -> scale
    scale_mat[:, 0, 0] = scale * half_width
    scale_mat[:, 1, 1] = scale * half_height
    scale_mat[:, 0, 2] = translation_factor[:, 0] * width * scale
    scale_mat[:, 1, 2] = translation_factor[:, 1] * height * scale
    scale_mat[:, 2, 2] = 1

    # rotate -> transform back coordinates
    rotate_sin = torch.sin(rotate)
    rotate_cos = torch.cos(rotate)
    rotate_mat = torch.zeros(batch_size, 3, 3).to(x.device, non_blocking=True)
    rotate_mat[:, 0, 0] = rotate_cos / half_width
    rotate_mat[:, 0, 1] = -rotate_sin / half_width
    rotate_mat[:, 0, 2] = 0
    rotate_mat[:, 1, 0] = rotate_sin / half_height
    rotate_mat[:, 1, 1] = rotate_cos / half_height
    rotate_mat[:, 1, 2] = 0
    rotate_mat[:, 2, 2] = 1

    theta = torch.bmm(rotate_mat, scale_mat)
    theta = theta.inverse()[:, :2]
    grid = affine_grid(theta, x.size())
    x = grid_sample(x, grid)

    if blur_sigma is None:
        return x

    kernel_halfsize = int((blur_sigma * 3).ceil().max())
    kernel_size = kernel_halfsize * 2 + 1
    ky = torch.arange(-kernel_halfsize, kernel_halfsize + 1).to(x.device, dtype=torch.float, non_blocking=True)
    kx = torch.arange(-kernel_halfsize, kernel_halfsize + 1).to(x.device, dtype=torch.float, non_blocking=True)
    field = torch.stack([kx.expand(kernel_size, -1), ky[:, None].expand(-1, kernel_size)], dim=2)
    kernel = torch.exp(- field.pow(2).sum(dim=2).view(1, kernel_size, kernel_size) / 2 / blur_sigma.pow(2).view(-1, 1, 1))
    kernel /= kernel.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)
    flat_channels = batch_size * in_channels
    kernel = kernel.view(batch_size, 1, kernel_size, kernel_size).repeat(1, in_channels, 1, 1).view(flat_channels, 1, kernel_size, kernel_size)
    x = F.conv2d(x.view(1, flat_channels, height, width), kernel, padding=(kernel_halfsize, kernel_halfsize), groups=flat_channels).view(batch_size, in_channels, height, width)

    return x
