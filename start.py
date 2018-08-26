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
from io import StringIO

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

from pose.utils.evaluation import AverageMeter
from utils.miscs import mkdir_p
from utils.globals import config, hparams, globalvars
from utils.miscs import ask, is_main_process
from utils.checkpoint import detect_checkpoint, save_pred
from utils.log import log_w, log_e, log_q, log_i, log_suc, log_progress

from experiments.baseexperiment import BaseExperiment, EpochContext

global enable_sigint_handler
def init_sigint_handler():
    global enable_sigint_handler
    # Handle sigint
    globalvars.sigint_triggered = False
    def _enable_sigint_handler():
        ori_sigint_handler = signal.getsignal(signal.SIGINT)
        def sigint_handler(sig, frame):
            if globalvars.sigint_triggered:
                ori_sigint_handler(sig, frame)
            globalvars.sigint_triggered = True
            if is_main_process():
                log_i("SIGINT DETECTED")
        signal.signal(signal.SIGINT, sigint_handler)
    enable_sigint_handler = _enable_sigint_handler

def check_hparams_consistency(old_hparams, new_hparams, ignore_hparams_mismatch=False):
    def safe_yaml_convert(rt_yaml):
        sio = StringIO()
        YAML().dump(rt_yaml, sio)
        return YAML(typ="safe").load(sio.getvalue())

    if safe_yaml_convert(old_hparams) != safe_yaml_convert(new_hparams):
        log_w("hparams from config and from checkpoint are not equal")
        if not ignore_hparams_mismatch:
            print("In current:")
            YAML().dump(new_hparams, sys.stdout)
            print("<<<<\n>>>>\nIn checkpoint:")
            YAML().dump(old_hparams, sys.stdout)
            if not ask("Continue?"):
                return False
        else:
            log_i("hparams mismatch ignored")

    return True

def override_hparams(_hparams, override):
    def set_hierarchic_attr(var, var_name_hierarchic, var_value):
        if len(var_name_hierarchic) > 1:
            set_hierarchic_attr(var[var_name_hierarchic[0]], var_name_hierarchic[1:], var_value)
        else:
            var[var_name_hierarchic[0]] = var_value
    for var_name, var_value in override:
        set_hierarchic_attr(_hparams, var_name.split("."), eval(var_value))

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

def main(args, unknown_args):
    init_matplotlib_backend()
    init_sigint_handler()

    init_config(args.CONF, args.config)
    exp_module_name = args.EXP
    hparams.update(get_hparams(exp_module_name))
    exp_name = hparams["name"]
    before_epoch = hparams["before_epoch"]
    globalvars.exp_name = exp_name

    # Override config from command line
    if args.override is not None:
        override_hparams(hparams, args.override)

    # Substitude var in configs
    config.checkpoint = config.checkpoint.format(**{"exp": exp_name, "id": hparams["id"]})
    if config.resume is not None:
        config.resume = config.resume.format(**{"exp": exp_name, "id": hparams["id"]})

    # Check if checkpoint existed
    if config.resume is None and detect_checkpoint(checkpoint_folder=config.checkpoint):
        log_w("Exist files in {}".format(config.checkpoint))
        if not ask("Do you want to delete ALL files in {}?".format(config.checkpoint),
                   posstr="yes", negstr="n"):
            log_q("Will not delete")
            sys.exit(0)
        else:
            log_i("Deleting " + config.checkpoint)
            shutil.rmtree(config.checkpoint)

    if not os.path.isdir(config.checkpoint):
        mkdir_p(config.checkpoint)

    # Create experiment
    log_progress("Creating model")

    exp_module = importlib.import_module("experiments." + exp_module_name)
    assert issubclass(exp_module.Experiment, BaseExperiment), exp_module_name + ".Experiment is not a subclass of BaseExperiment"
    exp = exp_module.Experiment()

    if config.resume is None:
        hparams_cp_file = os.path.join(config.checkpoint, "hparams.yaml")
        with open(hparams_cp_file, "w") as f:
            YAML().dump(hparams, f)
    else:
        # Load checkpoint
        if args.resume_file is None:
            log_e("Please specify resume_file in the argumen")
            sys.exit(1)

        resume_full = os.path.join(config.resume, args.resume_file)
        if not os.path.isfile(resume_full):
            log_e("No checkpoint found at '{}'".format(config.resume))
            sys.exit(1)

        # Check hparams consistency
        old_hparams_file = os.path.join(config.resume, "hparams.yaml")
        if os.path.isfile(old_hparams_file):
            with open(old_hparams_file, "r") as f:
                old_hparams = YAML().load(f)
            if not check_hparams_consistency(old_hparams, hparams, ignore_hparams_mismatch=args.ignore_hparams_mismatch):
                log_q("hparams mismatch")
                sys.exit(0)
        else:
            log_w("No hparams detected in resume folder {}".format(config.resume))

        log_progress("Loading checkpoint '{}'".format(resume_full))
        before_epoch = exp.load_checkpoint(config.resume, args.resume_file,
                                           no_strict_model_load=config.no_strict_model_load,
                                           no_criterion_load=config.no_criterion_load,
                                           no_optimizer_load=config.no_optimizer_load)
        if before_epoch is None:
            log_q("Failed loading checkpoint")
            sys.exit(1)

        log_progress("Loaded checkpoint (epoch {})".format(before_epoch))

        if not config.evaluate:
            check_future_checkpoint(config.checkpoint, before_epoch + 1)

    log_progress("Initiating dataloader")
    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset,
        collate_fn=exp.train_collate_fn,
        batch_size=hparams["train_batch"],
        num_workers=config.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=exp.train_drop_last,
        worker_init_fn=exp.worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset,
        collate_fn=exp.valid_collate_fn,
        batch_size=hparams["test_batch"],
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=exp.worker_init_fn)

    if args.run is not None:
        run_name = args.run
        run_path = os.path.join("runs", run_name)
        if not os.path.isdir(run_path):
            log_e("Run not found")
            sys.exit(1)
        purge_step = before_epoch * len(train_loader)
        log_w("Will purge run {} step from {}".format(run_name, purge_step))
        if not ask("Confirm?", posstr="yes", negstr="n"):
            log_q("No purging")
            sys.exit(0)
        globalvars.tb_writer = SummaryWriter(log_dir=run_path, purge_step=purge_step)
    else:
        run_name = "{exp_name}_{exp_id}/{datetime}".format(
            exp_name=exp_name,
            exp_id=hparams["id"],
            datetime=datetime.datetime.now().strftime("%b%d_%H-%M-%S"))
        run_path = "runs/" + run_name
        globalvars.tb_writer = SummaryWriter(log_dir=run_path)

    if config.handle_sig:
        enable_sigint_handler()

    if config.evaluate:
        print()
        log_i("Evaluation-only mode")
        result_collection = validate(val_loader, exp, 0, 0)
        if result_collection is not None:
            print()
            save_pred(result_collection, checkpoint_folder=config.checkpoint, pred_file="pred_evaluate.npy")
    else:
        train_eval_loop(exp, before_epoch + 1, hparams["epochs"] + 1, train_loader, val_loader)

    log_suc("Run finished!")
    if not ask("Save current run?", posstr="y", negstr="delete", ansretry=False, ansdefault=True):
        log_i("Deleting current run")
        shutil.rmtree(run_path)

def train_eval_loop(exp, start_epoch, stop_epoch, train_loader, val_loader):
    cur_step = len(train_loader) * (start_epoch - 1)
    for epoch in range(start_epoch, stop_epoch):
        exp.epoch_start(epoch, cur_step)

        print()
        log_progress("Epoch: %d | LR: %.8f" % (epoch, exp.cur_lr))

        # train for one epoch
        train(train_loader, exp, epoch, cur_step, em_valid_int=config.em_valid_int, val_loader=val_loader)

        if globalvars.sigint_triggered:
            return False

        cur_step += len(train_loader)

        if exp.use_post:
            log_progress("Post train:")
            post_train(train_loader, exp, epoch, cur_step)

        if globalvars.sigint_triggered:
            return False

        # evaluate on validation set
        if config.skip_val > 0 and epoch % config.skip_val == 0:
            log_progress("Validation:")
            result_collection = validate(val_loader, exp, epoch, cur_step)
        else:
            log_progress("Skip validation")
            result_collection = None

        if globalvars.sigint_triggered:
            return False

        cp_filename = "checkpoint_{}.pth.tar".format(epoch)
        exp.save_checkpoint(config.checkpoint, cp_filename, epoch)

        if result_collection is not None:
            preds_filename = "preds_{}.npy".format(epoch)
            save_pred(result_collection, checkpoint_folder=config.checkpoint, pred_file=preds_filename)

        exp.epoch_end(epoch)

        if globalvars.sigint_triggered:
            return False

    return True

def train(train_loader, exp, epoch, cur_step, em_valid_int=0, val_loader=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ctx = EpochContext()

    # switch to train mode
    exp.model.train()

    end = time.time()
    start_time = time.time()

    iter_length = len(train_loader)

    with torch.autograd.enable_grad():
        for i, batch in enumerate(train_loader):
            em_valid = False
            if em_valid_int > 0 and (i+1) % em_valid_int == 0 and iter_length - (i+1) >= max(em_valid_int/2, 1):
                em_valid = True

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

            exp.iter_step(loss, progress=progress)

            loss = loss.item() if loss is not None else None

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

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
            exp.print_iter(epoch_ctx)
            exp.summary_scalar(epoch_ctx, cur_step, "train")
            
            if globalvars.sigint_triggered:
                break

            if em_valid:
                print()
                log_progress("Embeded Validation:")
                validate(val_loader, exp, epoch, cur_step + 1, store_result=True)
                print()
                log_progress("Return to train:")
                exp.model.train()
                end = time.time()

            if globalvars.sigint_triggered:
                break

            if config.fast_pass_train > 0 and (i+1) >= config.fast_pass_train:
                log_progress("Fast Pass!")
                break

            cur_step += 1

def post_train(train_loader, exp, epoch, cur_step):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ctx = EpochContext()

    # switch to train mode
    exp.model.train()

    end = time.time()
    start_time = time.time()

    iter_length = len(train_loader)

    with torch.autograd.enable_grad():
        for i, batch in enumerate(train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            progress = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step,
                "train": True,
                "post": True
            }

            result = exp.iter_process(epoch_ctx, batch, progress=progress)
            loss = result["loss"]

            exp.iter_step(loss, progress=progress)

            loss = loss.item() if loss is not None else None

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

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
            exp.print_iter(epoch_ctx)
            exp.summary_scalar(epoch_ctx, cur_step+i, "post_train")

            if globalvars.sigint_triggered:
                break

            if globalvars.sigint_triggered:
                break

            if config.fast_pass_train > 0 and (i+1) >= config.fast_pass_train:
                log_progress("Fast Pass!")
                break

def combine_result(prev, cur):
    assert type(prev) == type(cur)
    if isinstance(prev, dict):
        assert set(prev.keys()) == set(cur.keys())
        for key in prev:
            prev[key] = combine_result(prev[key], cur[key])
        return prev
    elif isinstance(prev, list):
        return prev + cur
    elif isinstance(prev, torch.Tensor):
        assert prev.size()[1:] == cur.size()[1:]
        return torch.cat([prev, cur])
    elif isinstance(prev, np.ndarray):
        assert prev.shape[1:] == cur.shape[1:]
        return torch.concatenate([prev, cur])
    raise TypeError("Not supported type")        

def validate(val_loader, exp, epoch, cur_step, store_result=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_ctx = EpochContext()

    result_collection = None
    store_lim = config.store_lim
    if store_lim is None:
        store_lim = len(val_loader.dataset)

    preds = None

    # switch to evaluate mode
    exp.model.eval()

    end = time.time()
    start_time = time.time()
    iter_length = len(val_loader)
    data_counter = 0
    with torch.autograd.no_grad():
        for i, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            batch_size = len(batch["index"])

            progress = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step,
                "train": False
            }

            result = exp.iter_process(epoch_ctx, batch, progress=progress)

            if result["pred"] is not None:
                if preds is None:
                    preds = result["pred"]
                else:
                    preds = combine_result(preds, result["pred"])

            loss = result["loss"]
            
            index = result["index"] if "index" in result else None

            if index is None:
                index = list(range(data_counter, data_counter+batch_size))

            result_save = result["save"]
            if result_save is not None and store_result:
                if result_collection is None:
                    if isinstance(result_save, torch.Tensor):
                        result_collection = torch.zeros((store_lim,) + result_save.size()[1:])
                    elif isinstance(result_save, np.ndarray):
                        result_collection = np.zeros((store_lim,) + result_save.shape[1:])
                    elif isinstance(result_save, collections.Sequence):
                        result_collection = dict()
                    else:
                        raise TypeError("Not valid result_save type")

                for n in range(len(result_save)):
                    if index[n] >= store_lim:
                        continue
                    result_collection[index[n]] = result_save[n]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            data_counter += len(index)

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
            exp.print_iter(epoch_ctx)

            if globalvars.sigint_triggered:
                break

            if config.fast_pass_valid > 0 and (i+1) >= config.fast_pass_valid:
                log_progress("Fast Pass!")
                break

    exp.summary_scalar_avg(epoch_ctx, cur_step, phase="valid")
    if preds is not None:
        exp.evaluate(preds, cur_step)
    else:
        log_i("pred is None, skip evaluation")

    return result_collection

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("CONF", type=str)
    argp.add_argument("EXP", type=str)
    argp.add_argument("-r", dest="resume_file", type=str)
    argp.add_argument("--ptvsd", action="store_true")
    argp.add_argument("--ignore-hparams-mismatch", action="store_true")
    argp.add_argument("--run", type=str)
    argp.add_argument("--override", nargs=2, metavar=("var", "value"), action="append")
    argp.add_argument("--config", nargs=2, metavar=("var", "value"), action="append")
    return argp.parse_known_args()

def init_config(conf_name, config_override):
    with open("experiments/config.yaml", "r") as f:
        conf = YAML(typ="safe").load(f)
    conf_data = conf[conf_name]
    config.update(conf_data.items())
    if config_override:
        config.update(dict(map(lambda x: (x[0], eval(str(x[1]))), config_override)))

def get_hparams(exp_name, hp_file="experiments/hparams.yaml"):
    with open(hp_file, "r") as f:
        return YAML().load(f)[exp_name]

if __name__ == "__main__":
    _args, _unknown_args = get_args()
    if _args.ptvsd:
        import ptvsd
        import platform
        ptvsd.enable_attach("mydebug", address = ("0.0.0.0", 23456))
        if platform.node() == "lhtserver-2":
            print("Waiting for debugger...")
            ptvsd.wait_for_attach()
            print("Debugger attached!")

    # For now we do not need extra args
    assert len(_unknown_args) == 0, "Unknown args: " + str(_unknown_args)
    main(_args, _unknown_args)
