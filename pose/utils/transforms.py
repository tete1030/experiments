from __future__ import absolute_import

import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch

from .misc import *
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

def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
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
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

# Transform pixel location to different reference
def transform(pt, center, scale, res, invert=0, rot=0):
    assert type(pt) is np.ndarray 
    t = get_transform(center, scale, res, rot=rot)
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


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords

# deprecated with using of `warp`
def crop(img, center, scale, res, rot=0):
    assert type(img) is np.ndarray, type(img)
    assert img.dtype == np.float32, img.dtype

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    new_img = scipy.misc.imresize(new_img, res).astype(np.float32) / 255
    return new_img
