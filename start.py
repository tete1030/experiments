#!python3
import os
import sys
import argparse
import time
import importlib
import collections
import datetime
import signal
import re
import shutil
import unittest

def init_matplotlib_backend():
    from utils.log import log_w
    # Handle matplotlib backend error when DISPLAY is wrong
    # the error originates from Tk used in matplotlib
    if matplotlib.get_backend() != "module://ipykernel.pylab.backend_inline":
        try:
            import tkinter
            tkinter.Tk().destroy()
            matplotlib_backend = "TkAgg"
        except tkinter.TclError:
            log_w("Cannot use TkAgg for matplotlib, using Agg")
            matplotlib_backend = "Agg"
        else:
            del tkinter
        matplotlib.use(matplotlib_backend)
        del matplotlib_backend

import matplotlib; init_matplotlib_backend()
import torch
import torch.nn.parallel
import torch.optim
import numpy as np
from ruamel.yaml import YAML
from tensorboardX import SummaryWriter
from deepdiff import DeepDiff
from pprint import pprint

from utils.globals import config, hparams, globalvars
from utils.miscs import mkdir_p, ask, is_main_process, wait_key, safe_yaml_convert, dict_toupper, YAMLScopeError, set_yaml_scope, dump_yaml
from utils.checkpoint import detect_checkpoint
from utils.log import log_w, log_e, log_q, log_i, log_suc, log_progress
from utils.train import TrainContext, ValidContext
from lib.utils.evaluation import AverageMeter
from experiments.baseexperiment import BaseExperiment, EpochContext

globalvars.main_context.sigint_triggered = False
def enable_sigint_handler():
    ori_sigint_handler = signal.getsignal(signal.SIGINT)
    def sigint_handler(sig, frame):
        if globalvars.main_context.sigint_triggered:
            ori_sigint_handler(sig, frame)
        globalvars.main_context.sigint_triggered = True
        if is_main_process():
            log_i("SIGINT DETECTED")
    signal.signal(signal.SIGINT, sigint_handler)

class ExitWithDelete(SystemExit):
    pass

def check_hparams_consistency(checkpoint_dir, new_hparams):
    # Check hparams consistency
    old_hparams_file = os.path.join(checkpoint_dir, "hparams.yaml")
    if os.path.isfile(old_hparams_file):
        with open(old_hparams_file, "r") as f:
            old_hparams = YAML().load(f)
        ddiff = DeepDiff(safe_yaml_convert(old_hparams), safe_yaml_convert(new_hparams))
        if ddiff:
            log_w("hparams are different:")
            if not config.ignore_hparams_mismatch:
                pprint(ddiff)
                if not ask("Continue loading?"):
                    log_q("Exit due to mismatched hparams")
                    raise ExitWithDelete(0)
            else:
                log_i("hparams mismatch ignored")
    else:
        log_w("No hparams detected in checkpoint folder {}".format(checkpoint_dir))

def check_future_checkpoint(checkpoint_folder, epoch_future_start):
    # Check if checkpoints of following epochs exist
    CP_RE = re.compile(r"^checkpoint_(\d+)\.pth\.tar$")
    def _is_future_checkpoint(x):
        match = CP_RE.match(x)
        return match is not None and int(match.group(1)) >= epoch_future_start
    future_checkpoints = detect_checkpoint(checkpoint_folder=checkpoint_folder,
                                           checkpoint_file=_is_future_checkpoint,
                                           return_files=True)
    if len(future_checkpoints) > 0:
        log_w("Exist future checkpoints in {}:".format(checkpoint_folder))
        list(map(lambda x: print("\t" + x), future_checkpoints))
        if not ask("Do you want to delete them?", posstr="yes", negstr="n"):
            print("No deletion")
            return False
        else:
            log_progress("Deleting future checkpoints in " + checkpoint_folder)
            for fcp in future_checkpoints:
                print("\tDeleting " + fcp)
                os.remove(os.path.join(checkpoint_folder, fcp))
    return True

def init_hparams(exp_module_name, hparams_override_list):
    hp_file="experiments/{}/hparams.yaml".format(exp_module_name)
    with open(hp_file, "r") as f:
        loaded_hparams = YAML().load(f)
    # Override config from command line
    if hparams_override_list:
        for override_key, override_value in hparams_override_list:
            new_value_parsed = YAML(typ="safe").load(override_value)
            original_value = set_yaml_scope(loaded_hparams, override_key, new_value_parsed)
            log_i("Overriding hparams item '{}' from '{}' to '{}'".format(override_key, original_value, new_value_parsed))

    globalvars.main_context.loaded_hparams = loaded_hparams
    _hparams = safe_yaml_convert(loaded_hparams)
    if "config" in _hparams:
        del _hparams["config"]
    hparams.update(dict_toupper(_hparams))

def init_config(conf_name, config_override_list):
    with open("experiments/config.yaml", "r") as f:
        conf = YAML(typ="safe").load(f)
    loaded_config = conf[conf_name]

    if "config" in globalvars.main_context.loaded_hparams:
        hparams_config = safe_yaml_convert(globalvars.main_context.loaded_hparams["config"])
        log_i("Overriding config from hparams:\n  " + "\n  ".join(dump_yaml(hparams_config).getvalue().strip(" \n\r").split("\n")))
        loaded_config.update(hparams_config)

    if config_override_list:
        for override_key, override_value in config_override_list:
            new_value_parsed = YAML(typ="safe").load(override_value)
            original_value = set_yaml_scope(loaded_config, override_key, new_value_parsed)
            log_i("Overriding config item '{}' from '{}' to '{}'".format(override_key, original_value, new_value_parsed))
    config.update(loaded_config)

def init_run(run_id):
    exp_name=hparams.EXP.NAME
    exp_id=hparams.EXP.ID

    if run_id is None:
        run_id = datetime.datetime.now().strftime("%b%d_%H%M%S")

    checkpoint_dir = config.checkpoint_dir_template.format(
        exp_name=exp_name,
        exp_id=exp_id,
        run_id=run_id)
    run_dir = config.run_dir_template.format(
        exp_name=exp_name,
        exp_id=exp_id,
        run_id=run_id)
    checkpoint_trash_dest = config.checkpoint_trash_dest_template.format(
        exp_name=exp_name,
        exp_id=exp_id,
        run_id=run_id)
    run_trash_dest = config.run_trash_dest_template.format(
        exp_name=exp_name,
        exp_id=exp_id,
        run_id=run_id)

    globalvars.main_context.run_id = run_id
    globalvars.main_context.checkpoint_dir = checkpoint_dir
    globalvars.main_context.run_dir = run_dir
    globalvars.main_context.checkpoint_trash_dest = checkpoint_trash_dest
    globalvars.main_context.run_trash_dest = run_trash_dest

def prepare_checkpoint_dir(resume_run_id):
    checkpoint_dir = globalvars.main_context.checkpoint_dir
    # Check if checkpoint existed
    if resume_run_id is None and detect_checkpoint(checkpoint_folder=checkpoint_dir, checkpoint_file=True):
        log_e("Files exist in checkpoint directory {}".format(checkpoint_dir))
        raise sys.exit(1)

    if not os.path.isdir(checkpoint_dir):
        mkdir_p(checkpoint_dir)

def init_exp(exp_module_name):
    # Create experiment
    log_progress("Creating model")
    exp_module = importlib.import_module("experiments." + exp_module_name)
    assert issubclass(exp_module.Experiment, BaseExperiment), exp_module_name + ".Experiment is not a subclass of BaseExperiment"
    exp = exp_module.Experiment()
    globalvars.main_context.exp = exp
    globalvars.main_context.num_train_iters = len(exp.train_loader)
    return exp

def init_tensorboard(run_dir, purge_step=None):
    log_i("Run dir: {}".format(run_dir))
    if purge_step is not None:
        if not os.path.isdir(run_dir):
            log_e("Run not found")
            raise ExitWithDelete(1)
        log_w("Will purge current run from step {}".format(purge_step))
        if not ask("Confirm?", posstr="yes", negstr="n"):
            log_q("No purging")
            raise ExitWithDelete(0)
        globalvars.main_context.tb_writer = SummaryWriter(log_dir=run_dir, purge_step=purge_step)
    else:
        globalvars.main_context.tb_writer = SummaryWriter(log_dir=run_dir)
        wait_key()
    return globalvars.main_context.tb_writer

def sanity_check(args):
    if not config.resume and args.run_id is not None:
        log_e("run_id should not be specified when not resuming")
        raise ExitWithDelete(1)

    if config.evaluate and not config.resume:
        log_e("config.resume should be set in evaluation-only mode")
        raise ExitWithDelete(1)

    if config.evaluate and config.use_tensorboard:
        log_e("Tensorboard should not be used in evaluation-only mode")
        raise ExitWithDelete(1)

def init(args):
    exp_module_name = args.EXP

    init_hparams(exp_module_name, args.hparams)
    init_config(args.CONF, args.config)
    sanity_check(args)

    init_run(args.run_id)

    exp = init_exp(exp_module_name=exp_module_name)

    return exp

def resume_checkpoint(exp, resume_run_id, cp_file, cp_epoch):
    is_local_resume = resume_run_id is not None
    if is_local_resume:
        resume_dir = config.checkpoint_dir_template.format(exp_name=hparams.EXP.NAME, exp_id=hparams.EXP.ID, run_id=resume_run_id)
        if cp_epoch is not None:
            resume_filepath = os.path.join(resume_dir, "checkpoint_{}.pth.tar".format(cp_epoch))
        elif cp_file is not None:
            resume_filepath = os.path.join(resume_dir, cp_file)
        else:
            log_e("Please specify checkpoint epoch or file for resuming")
            raise ExitWithDelete(1)
    else:
        if cp_file is None:
            log_e("Please specify full checkpoint path for resuming")
            raise ExitWithDelete(1)
        resume_dir = os.path.dirname(cp_file)
        resume_filepath = cp_file

    if not os.path.isfile(resume_filepath):
        log_e("No checkpoint found at '{}'".format(resume_filepath))
        raise ExitWithDelete(1)

    check_hparams_consistency(resume_dir, globalvars.main_context.loaded_hparams)

    log_progress("Loading checkpoint '{}'".format(resume_filepath))
    before_epoch = exp.load_checkpoint(resume_filepath,
        no_strict_model_load=config.no_strict_model_load,
        no_criterion_load=config.no_criterion_load,
        no_optimizer_load=config.no_optimizer_load)

    if before_epoch is None:
        log_e("Failed loading checkpoint")
        raise ExitWithDelete(1)

    log_progress("Loaded checkpoint (epoch {})".format(before_epoch))

    if is_local_resume and not config.evaluate:
        check_future_checkpoint(resume_dir, before_epoch + 1)

    return before_epoch

def dump_hparams(checkpoint_dir, loaded_hparams):
    hparams_cp_file = os.path.join(checkpoint_dir, "hparams.yaml")
    with open(hparams_cp_file, "w") as f:
        YAML().dump(loaded_hparams, f)

def cleanup(resume_run_id, caught_exception, exit_exception):
    try:
        globalvars.main_context.run_dir
        globalvars.main_context.checkpoint_dir
    except AttributeError:
        print("No run or checkpoint files created")
    else:
        do_trash = False
        if resume_run_id is None:
            if isinstance(exit_exception, ExitWithDelete):
                do_trash = True
            elif not ask("Save run and checkpoint?", posstr="y", negstr="trash", ansretry=2, ansdefault=True, timeout_sec=60 if caught_exception is None else None):
                do_trash = True

        if do_trash:
            log_i("Trashing run")
            if os.path.exists(globalvars.main_context.run_dir):
                if not os.path.isdir(globalvars.main_context.run_trash_dest):
                    os.makedirs(globalvars.main_context.run_trash_dest)
                shutil.move(globalvars.main_context.run_dir, globalvars.main_context.run_trash_dest)
            else:
                log_i("Run dir do not exist")
            log_i("Trashing checkpoint")
            if os.path.realpath(globalvars.main_context.run_dir) != os.path.realpath(globalvars.main_context.checkpoint_dir):
                if os.path.exists(globalvars.main_context.checkpoint_dir):
                    if not os.path.isdir(globalvars.main_context.checkpoint_trash_dest):
                        os.makedirs(globalvars.main_context.checkpoint_trash_dest)
                    shutil.move(globalvars.main_context.checkpoint_dir, globalvars.main_context.checkpoint_trash_dest)
                else:
                    log_i("Checkpoint dir do not exist")
        else:
            log_i("Run and checkpoint {} saved".format(globalvars.main_context.run_id))

    if caught_exception:
        import traceback
        wait_key(tip="Press any key to print exception...")
        print(''.join(traceback.format_exception(etype=type(caught_exception), value=caught_exception, tb=caught_exception.__traceback__)))
        if wait_key(tip="Press `d` to debug or other keys to exit...").lower() in ["d", ""]:
            import ipdb; ipdb.post_mortem(caught_exception.__traceback__)

def main(args):
    caught_exception = None
    exit_exception = None
    try:
        exp = init(args)

        if not config.resume:
            before_epoch = 0
        else:
            before_epoch = resume_checkpoint(exp, resume_run_id=args.run_id, cp_file=args.cp_file, cp_epoch=args.cp_epoch)

        prepare_checkpoint_dir(resume_run_id=args.run_id)
        if args.run_id is None:
            dump_hparams(globalvars.main_context.checkpoint_dir, globalvars.main_context.loaded_hparams)

        if config.use_tensorboard:
            if args.run_id:
                purge_step = before_epoch * globalvars.main_context.num_train_iters
            else:
                purge_step = None
            init_tensorboard(run_dir=globalvars.main_context.run_dir, purge_step=purge_step)

        if config.handle_sig:
            enable_sigint_handler()

        if config.evaluate:
            print()
            log_i("Evaluation-only mode")
            # Set current state to the last iteration of the last epoch
            exp.epoch_start(before_epoch, globalvars.main_context.num_train_iters * (before_epoch - 1), True)
            cur_step = globalvars.main_context.num_train_iters * before_epoch - 1
            validate(exp, before_epoch, cur_step, call_store=True)
            exp.epoch_end(before_epoch, cur_step, True)
        else:
            train_eval_loop(exp, before_epoch + 1, hparams.TRAIN.NUM_EPOCH + 1)

        log_suc("Run finished!")
    except KeyboardInterrupt:
        pass
    except SystemExit as e:
        exit_exception = e
    except Exception as e:
        log_e("Exception caught!")
        caught_exception = e

    cleanup(args.run_id, caught_exception, exit_exception)
    if exit_exception:
        sys.exit(exit_exception.code)

def train_eval_loop(exp, start_epoch, stop_epoch):
    cur_step = globalvars.main_context.num_train_iters * (start_epoch - 1)
    for epoch in range(start_epoch, stop_epoch):
        print()
        log_progress("Epoch: %d" % (epoch,))

        exp.epoch_start(epoch, cur_step, False)
        log_progress("Training:")

        if globalvars.main_context.sigint_triggered:
            return False

        # train for one epoch
        for cur_pause_step, is_final in train(exp, epoch, cur_step, pause_interval=config.valid_interval):
            print()
            log_progress("Validation:")
            validate(exp, epoch, cur_pause_step, call_store=is_final)
            if globalvars.main_context.sigint_triggered:
                break
            if not is_final:
                print()
                log_progress("Training:")

        if globalvars.main_context.sigint_triggered:
            return False

        cur_step += globalvars.main_context.num_train_iters
        exp.epoch_end(epoch, cur_step, False)

        if globalvars.main_context.sigint_triggered:
            return False

        cp_filename = "checkpoint_{}.pth.tar".format(epoch)
        if config.save_checkpoint:
            exp.save_checkpoint(os.path.join(globalvars.main_context.checkpoint_dir, cp_filename), epoch)

        if globalvars.main_context.sigint_triggered:
            return False

    return True

def train(exp:BaseExperiment, epoch:int, cur_step:int, pause_interval:int=0):
    if config.fast_pass_train == 0:
        log_progress("Fast Pass!")
        return

    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ctx = EpochContext()

    end = time.time()
    start_time = time.time()

    iter_length = globalvars.main_context.num_train_iters

    with torch.autograd.enable_grad(), TrainContext(exp.model):
        for i, batch in enumerate(exp.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            progress = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step,
                "train": True
            }

            result = exp.iter_process(epoch_ctx, batch, progress=progress)
            loss = result["loss"]

            exp.iter_step(epoch_ctx, loss, progress=progress)

            loss = loss.item() if loss is not None else None

            exp.summary_scalar(epoch_ctx, epoch, cur_step, "train")

            # measure elapsed time
            batch_time.update(time.time() - end)

            loginfo = ("{epoch:3}: ({batch:0{size_width}}/{size}) data: {data:.6f}s | batch: {bt:.3f}s | total: {total:3.1f}s").format(
                epoch=epoch,
                batch=i + 1,
                size_width=len(str(iter_length)),
                size=iter_length,
                data=data_time.val,
                bt=batch_time.val,
                total=time.time() - start_time,
            )
            print(loginfo, end="")
            exp.print_iter(epoch_ctx, epoch, cur_step)

            if globalvars.main_context.sigint_triggered:
                break

            if config.fast_pass_train is not None and (i+1) >= config.fast_pass_train:
                log_progress("Fast Pass!")
                yield cur_step, True
                break

            is_final = (i == iter_length - 1)
            if is_final or (pause_interval > 0 and (i+1) % pause_interval == 0 and iter_length - i - 1 > pause_interval / 2):
                yield cur_step, is_final

            cur_step += 1

            end = time.time()

def validate(exp:BaseExperiment, epoch:int, cur_step:int, call_store:bool) -> None:
    if config.fast_pass_valid == 0:
        log_progress("Fast Pass!")
        return

    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ctx = EpochContext()
    iter_length = len(exp.val_loader)

    end = time.time()
    start_time = time.time()

    with torch.autograd.no_grad(), ValidContext(exp.model):
        for i, batch in enumerate(exp.val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            progress = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step,
                "train": False
            }
            result = exp.iter_process(epoch_ctx, batch, progress=progress)

            # measure elapsed time
            batch_time.update(time.time() - end)

            loginfo = ("{epoch:3}: ({batch:0{size_width}}/{size}) data: {data:.6f}s | batch: {bt:.3f}s | total: {total:3.1f}s").format(
                epoch=epoch,
                batch=i + 1,
                size_width=len(str(iter_length)),
                size=iter_length,
                data=data_time.val,
                bt=batch_time.val,
                total=time.time() - start_time
            )
            print(loginfo, end="")
            exp.print_iter(epoch_ctx, epoch, cur_step)

            if globalvars.main_context.sigint_triggered:
                break

            if config.fast_pass_valid is not None and (i+1) >= config.fast_pass_valid:
                log_progress("Fast Pass!")
                break

            end = time.time()

    if config.use_tensorboard:
        exp.summary_scalar_avg(epoch_ctx, epoch, cur_step, phase="valid")

    exp.evaluate(epoch_ctx, epoch, cur_step)

    if call_store:
        exp.process_stored(epoch_ctx, epoch, cur_step)

def get_args(args=None):
    argp = argparse.ArgumentParser()
    argp.add_argument("CONF", type=str, nargs="?", default="default")
    argp.add_argument("EXP", type=str, nargs="?", default="main")
    argp.add_argument("-i", "--run-id", type=str)
    argp_cp_group = argp.add_mutually_exclusive_group()
    argp_cp_group.add_argument("-e", "--cp-epoch", type=int, default=None)
    argp_cp_group.add_argument("-f", "--cp-file", type=str, default=None)
    argp.add_argument("--ptvsd", action="store_true")
    argp.add_argument("-p", "--hparams", nargs=2, metavar=("var", "value"), action="append")
    argp.add_argument("-c", "--config", nargs=2, metavar=("var", "value"), action="append")
    if args is None:
        return argp.parse_args()
    else:
        return argp.parse_args(args)

if __name__ == "__main__":
    _args = get_args()
    if _args.ptvsd:
        import ptvsd
        import platform
        ptvsd.enable_attach("mydebug", address = ("0.0.0.0", 23456))
        print("Waiting for debugger...")
        ptvsd.wait_for_attach()
        print("Debugger attached!")

    main(_args)
