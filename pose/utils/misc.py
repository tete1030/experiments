from __future__ import absolute_import

import os
import shutil
import torch 
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def detect_checkpoint(checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    return os.path.isfile(os.path.join(checkpoint, filename)) or \
           os.path.isfile(os.path.join(checkpoint, 'model_best.pth.tar'))

def save_pred(preds, is_best=False, checkpoint='checkpoint', filename='preds_valid.npy'):
    preds_filepath = os.path.join(checkpoint, filename)
    np.save(preds_filepath, preds)
    if is_best:
        shutil.copyfile(preds_filepath, os.path.join(checkpoint, 'preds_best.npy'))
    return filename


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    try:
        lr_deg = schedule.index(epoch) + 1
    except ValueError:
        lr_deg = 0
    if lr_deg > 0:
        lr *= gamma ** lr_deg
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma ** lr_deg

    return lr
