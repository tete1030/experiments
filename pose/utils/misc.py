from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        if preds:
            shutil.copyfile(preds_filepath, os.path.join(checkpoint, 'preds_best.npy'))

def detect_checkpoint(checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    return os.path.isfile(os.path.join(checkpoint, filename)) or \
           os.path.isfile(os.path.join(checkpoint, 'model_best.pth.tar'))

def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.npy'):
    preds = to_numpy(preds)
    filename = os.path.join(checkpoint, filename)
    np.save(filename, preds)
    return filename


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr
