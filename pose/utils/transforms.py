import numpy as np
import torch

from .imutils import *


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x

def fliplr_map(x, flipIndex):
    """
    flip output map
    """
    assert type(x) is np.ndarray
    assert x.ndim in [3, 4]

    # Change left-right parts
    newx = x[...,flipIndex,:,:]
    newx = fliplr_chwimg(newx)

    return newx

def fliplr_pts(x, flipIndex, width):
    """
    flip coords
    """
    assert type(x) is np.ndarray
    assert x.ndim in [2, 3]

    assert x.shape[-2] == len(flipIndex)

    newx = x[..., flipIndex, :]

    # Flip horizontal
    newx[..., 0] = (width - newx[..., 0]).astype(newx.dtype)

    return newx

def fliplr_chwimg(x):
    assert isinstance(x, np.ndarray)
    assert x.ndim in [2, 3, 4]
    return x[..., ::-1].copy()

# python3-fized
def get_transform(center, ori_size, new_size, rot=0, scale=None):
    """
    General image processing functions
    """
    # Generate transformation matrix
    assert (ori_size is not None and scale is None) or (ori_size is None and scale is not None)
    if isinstance(new_size, (int, float)):
        new_size = (new_size, new_size)
    assert len(new_size) == 2

    if ori_size is not None:
        if isinstance(ori_size, (int, float)):
            scale = (float(new_size[0]) / ori_size, float(new_size[1]) / ori_size)
        else:
            assert len(ori_size) == 2
            scale = (float(new_size[0]) / ori_size[0], float(new_size[1]) / ori_size[1])
    else:
        if isinstance(scale, (int, float)):
            scale = (float(scale), float(scale))
        assert len(scale) == 2

    # Ensure ori_size is not used
    del ori_size

    t = np.zeros((3, 3))
    t[0, 0] = scale[0]
    t[1, 1] = scale[1]
    # Origin at left-up conner
    t[0, 2] = -scale[0] * center[0] + float(new_size[0]) / 2
    t[1, 2] = -scale[1] * center[1] + float(new_size[1]) / 2
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -float(new_size[0]) / 2
        t_mat[1,2] = -float(new_size[1]) / 2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

# Transform pixel location to different reference
def transform(pt, center, ori_size, new_size, invert=0, rot=0, scale=None):
    assert type(pt) is np.ndarray 
    t = get_transform(center, ori_size, new_size, rot=rot, scale=scale)
    if invert:
        t = np.linalg.inv(t)

    ori_dtype = pt.dtype
    pt = pt.astype(float)
    assert pt.ndim == 1 or pt.ndim == 2
    if pt.ndim == 1:
        new_pt = np.array([pt[0], pt[1], 1.]).T
    elif pt.ndim == 2:
        new_pt = np.c_[pt[:, 0:2], np.ones((pt.shape[0], 1))].T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].T.astype(ori_dtype)


def transform_preds(coords, center, ori_size, new_size):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.from_numpy(transform(coords[p, 0:2].numpy(), center, ori_size, new_size))
    return coords
