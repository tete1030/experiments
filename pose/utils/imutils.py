import torch
import torch.nn as nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import cv2

# Generate heatmap quickly
# derived from https://github.com/umich-vl/pose-ae-train
class HeatmapGenerator(object):
    def __init__(self, out_res=None, sigma=None):
        """HeatmapGenerator
        
        Keyword Arguments:
            out_res {tuple or int} -- (w, h) or l (default: {None})
            sigma {float} -- map gaussian kernel size (default: {None})
        """

        self.out_res = out_res
        if sigma is not None:
            self.sigma = float(sigma)
            self.sigma3 = int(np.around(3 * self.sigma))
            self.g = self._gen_temp(self.sigma, self.sigma3)
        else:
            self.sigma = None
            self.sigma3 = None
            self.g = None

    @classmethod
    def _gen_temp(cls, sigma, sigma3):
        size = 2 * sigma3 + 3
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = sigma3 + 1, sigma3 + 1
        sigma = float(sigma)
        return np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, pt, index, out, sigma=None, normalize_factor=None, out_res=None, mingle_mode="max"):
        assert self.sigma is not None or sigma is not None
        if sigma is None:
            sigma = self.sigma
            sigma3 = self.sigma3
            g = self.g
        else:
            sigma = float(sigma)
            sigma3 = int(np.around(3 * sigma))
            g = self._gen_temp(sigma, sigma3)

        if normalize_factor is not None:
            g = g * float(normalize_factor)

        assert self.out_res is not None or out_res is not None
        if out_res is None:
            out_res = self.out_res

        assert isinstance(out_res, (int, tuple, np.ndarray))
        assert not isinstance(out_res, tuple) or len(out_res) == 2
        assert not isinstance(out_res, np.ndarray) or (out_res.shape == (2,) and out_res.dtype == np.int32)

        if isinstance(out_res, int):
            out_res = (out_res, out_res)
        else:
            out_res = (out_res[0], out_res[1])

        x, y = int(pt[0]), int(pt[1])
        if x < (-sigma3 - 1) or y < (-sigma3 - 1) or \
                x >= (out_res[0] + sigma3 + 1) or y >= (out_res[1] + sigma3 + 1):
            #print('not in', x, y)
            return
        ul = int(x - sigma3 - 1), int(y - sigma3 - 1)
        br = int(x + sigma3 + 2), int(y + sigma3 + 2)

        c,d = max(0, -ul[0]), min(br[0], out_res[0]) - ul[0]
        a,b = max(0, -ul[1]), min(br[1], out_res[1]) - ul[1]

        cc,dd = max(0, ul[0]), min(br[0], out_res[0])
        aa,bb = max(0, ul[1]), min(br[1], out_res[1])
        if mingle_mode == "max":
            out[index][aa:bb,cc:dd] = np.maximum(out[index][aa:bb,cc:dd], g[a:b,c:d])
        elif mingle_mode == "add":
            out[index][aa:bb,cc:dd] += g[a:b,c:d]
        else:
            raise ValueError("Unknown mingle_mode='%s'" % (mingle_mode,))

def batch_resize(im, new_shape):
    assert isinstance(new_shape, tuple) and len(new_shape) == 2 and isinstance(new_shape[0], int) and isinstance(new_shape[1], int)
    im_pre_shape = im.shape[:-2]
    im_post_shape = im.shape[-2:]
    if im_post_shape == new_shape:
        return im
    im = im.reshape((-1,) + im_post_shape)
    return np.array([cv2.resize(im[i], (new_shape[1], new_shape[0])) for i in range(im.shape[0])]).reshape(im_pre_shape + new_shape)
