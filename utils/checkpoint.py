import os
import shutil
import sys
import re
import torch
import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from utils.miscs import ask, wait_key
from utils.log import log_w, log_i, log_q
from copy import deepcopy

__all__ = ["save_checkpoint", "detect_checkpoint", "save_pred", "load_pretrained_loose"]

RE_TYPE = type(re.compile(""))

def save_checkpoint(state, checkpoint_folder, checkpoint_file, force_replace=False):
    filepath = os.path.join(checkpoint_folder, checkpoint_file)
    if os.path.exists(filepath):
        if os.path.isfile(filepath):
            log_w("Checkpoint file {} exists".format(filepath))
            if force_replace:
                log_i("Force replacing")
            elif not ask("Replace it?"):
                log_w("Skip saving {}".format(filepath))
                return
            os.remove(filepath)
        else:
            log_w("Checkpoint file {} exists as a folder".format(filepath))
            if force_replace:
                log_i("Force replacing")
            elif not ask("Delete it?"):
                log_w("Skip saving {}".format(filepath))
                return
            shutil.rmtree(filepath)
    torch.save(state, filepath)
    return filepath

def detect_checkpoint(checkpoint_folder, checkpoint_file=None, return_files=False):
    folder_exist = os.path.exists(checkpoint_folder)
    if checkpoint_file is None:
        return checkpoint_folder if return_files else folder_exist

    if isinstance(checkpoint_file, str):
        filepath = os.path.join(checkpoint_folder, checkpoint_file)
        return filepath if return_files else os.path.exists(filepath)
    elif isinstance(checkpoint_file, RE_TYPE) or callable(checkpoint_file):
        if isinstance(checkpoint_file, RE_TYPE):
            fn = checkpoint_file.match
        else:
            fn = checkpoint_file
        with os.scandir(checkpoint_folder) as fileiter:
            filteriter = filter(lambda x: fn(x.name), fileiter)
            if return_files:
                return list(map(lambda x: x.name, filteriter))
            else:
                return any(filteriter)
    else:
        assert False

def save_pred(preds, checkpoint_folder, pred_file, force_replace=False):
    preds_filepath = os.path.join(checkpoint_folder, pred_file)
    if os.path.exists(preds_filepath):
        if os.path.isfile(preds_filepath):
            log_w("Pred file {} exists".format(preds_filepath))
            if force_replace:
                log_i("Force replacing")
            elif not ask("Replace it?"):
                log_w("Skip saving {}".format(preds_filepath))
                return
            os.remove(preds_filepath)
        else:
            log_w("Pred file {} exists as a folder".format(preds_filepath))
            if force_replace:
                log_i("Force replacing")
            elif not ask("Delete it?"):
                log_w("Skip saving {}".format(preds_filepath))
                return
            shutil.rmtree(preds_filepath)
    np.save(preds_filepath, preds)
    return preds_filepath

class RejectLoadError(Exception):
    pass

def load_pretrained_loose(model_state_dict, pretrained_state_dict, pause_model_mismatch=True, confirm_model_size_mismatch=True, inplace=True):
    from collections import OrderedDict
    if not inplace:
        model_state_dict = deepcopy(model_state_dict)
    model_missing_keys = set(list(pretrained_state_dict.keys())) - set(list(model_state_dict.keys()))
    model_extra_keys = set(list(model_state_dict.keys())) - set(list(pretrained_state_dict.keys()))
    if len(model_missing_keys) > 0:
        log_w("Model missing keys: " + str(model_missing_keys))
    if len(model_extra_keys) > 0:
        log_w("Model extra keys: " + str(model_extra_keys))
    if pause_model_mismatch and (len(model_missing_keys) > 0 or len(model_extra_keys) > 0):
        wait_key()

    for k, v in pretrained_state_dict.items():
        if k in model_missing_keys:
            continue
        model_k_size = model_state_dict[k].size()
        pretr_k_size = v.size()
        if model_k_size != pretr_k_size and k.endswith(".weight") and len(model_k_size) == len(pretr_k_size) and len(model_k_size) == 4:
            # Output more than pretrained
            if model_k_size[0] > pretr_k_size[0]:
                log_w("Model output channel size ({}) larger than pretrained ({}) for {}".format(model_k_size[0], pretr_k_size[0], k))
                if confirm_model_size_mismatch and not ask("Would you like adaptive loading?"):
                    raise RejectLoadError()
            elif model_k_size[0] < pretr_k_size[0]:
                log_w("Model output channel size ({}) smaller than pretrained ({}) for {}".format(model_k_size[0], pretr_k_size[0], k))
                if confirm_model_size_mismatch and not ask("Would you like adaptive loading?"):
                    raise RejectLoadError()

            # Input more than pretrained
            if model_k_size[1] > pretr_k_size[1]:
                log_w("Model input channel size ({}) larger than pretrained ({}) for {}.".format(model_k_size[1], pretr_k_size[1], k))
                if confirm_model_size_mismatch and not ask("Would you like adaptive loading?"):
                    raise RejectLoadError()
            elif model_k_size[1] < pretr_k_size[1]:
                log_w("Model input channel size ({}) smaller than pretrained ({}) for {}.".format(model_k_size[1], pretr_k_size[1], k))
                if confirm_model_size_mismatch and not ask("Would you like adaptive loading?"):
                    raise RejectLoadError()

            min_output_channels = min(model_k_size[0], pretr_k_size[0])
            min_input_channels = min(model_k_size[1], pretr_k_size[1])

            model_state_dict[k].data[:min_output_channels, :min_input_channels] = v[:min_output_channels, :min_input_channels]
        else:
            model_state_dict[k] = v
    return model_state_dict
