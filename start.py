#!python3
import os
import sys
import argparse
import time

# Handle matplotlib backend error when DISPLAY is wrong
# the error originates from Tk used in matplotlib
import matplotlib
if matplotlib.get_backend() != "module://ipykernel.pylab.backend_inline":
    matplotlib_backend = "TkAgg"
    try:
        import tkinter
        tkinter.Tk().destroy()
    except tkinter.TclError:
        print("Cannot use TkAgg for matplotlib, using Agg")
        matplotlib_backend = "Agg"
    else:
        del tkinter
    matplotlib.use(matplotlib_backend)
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from pose.utils.evaluation import AverageMeter
from pose.utils.misc import save_checkpoint, detect_checkpoint, save_pred
from pose.utils.osutils import mkdir_p
import pose.utils.config as config

import numpy as np
import importlib
from ruamel.yaml import YAML
from io import StringIO

import collections
from tensorboardX import SummaryWriter
import datetime
from experiments.baseexperiment import BaseExperiment, EpochContext

# Handle sigint
import signal
from utils.miscs import is_main_process
config.sigint_triggered = False
def enable_sigint_handler():
    ori_sigint_handler = signal.getsignal(signal.SIGINT)
    def sigint_handler(signal, frame):
        if config.sigint_triggered:
            ori_sigint_handler(signal, frame)
        config.sigint_triggered = True
        if is_main_process():
            print("[SIGINT DETECTED]")
    signal.signal(signal.SIGINT, sigint_handler)

def ask(question, posstr="y", negstr="n", ansretry=False, ansdefault=False):
    if ansretry is False:
        ansretry = 1
    else:
        assert isinstance(ansretry, int)

    retry_count = 0
    while True:
        ans = input(question + " (%s|%s) :" % (posstr, negstr))
        retry_count += 1

        if ans == posstr:
            return True
        elif ans == negstr:
            return False
        else:
            if retry_count < ansretry:
                print("[Error] Illegal answer! Retry")
                continue
            else:
                print("[Error] Illegal answer!")
                return ansdefault

def check_hparams_consistency(old_hparams, new_hparams):
    def safe_yaml_convert(rt_yaml):
        sio = StringIO()
        YAML().dump(rt_yaml, sio)
        return YAML(typ="safe").load(sio.getvalue())

    if safe_yaml_convert(old_hparams) != safe_yaml_convert(new_hparams):
        return False
    return True

def main(args, unknown_args):
    init_config(args.CONF, args.config)
    exp_mod = args.EXP
    hparams = get_hparams(exp_mod)
    exp_name = hparams["name"]
    config.exp_name = exp_name

    # Override config from command line
    if args.override is not None:
        def set_hierarchic_attr(var, var_name_hierarchic, var_value):
            if len(var_name_hierarchic) > 1:
                set_hierarchic_attr(var[var_name_hierarchic[0]], var_name_hierarchic[1:], var_value)
            else:
                var[var_name_hierarchic[0]] = var_value
        for var_name, var_value in args.override:
            set_hierarchic_attr(hparams, var_name.split("."), eval(var_value))
        set_hierarchic_attr = None

    # Substitude var in configs
    config.checkpoint = config.checkpoint.format(**{"exp": exp_name, "id": hparams["id"]})
    if config.resume is not None:
        config.resume = config.resume.format(**{"exp": exp_name, "id": hparams["id"]})

    # Check if checkpoint existed
    hparams_cp_file = os.path.join(config.checkpoint, "hparams.yaml")
    if config.resume is None and ( \
            detect_checkpoint(checkpoint=config.checkpoint) or \
            os.path.isfile(hparams_cp_file)):
        if not ask("[Warning] Exist files in %s" % config.checkpoint + "\n" +
                "Do you want to delete files in %s ?" % config.checkpoint,
                posstr="yes", negstr="n"):
            print("[Exit] Will not delete")
            sys.exit(0)
        else:
            print("==> Deleting %s" % config.checkpoint)
            import shutil
            shutil.rmtree(config.checkpoint)

    if not os.path.isdir(config.checkpoint):
        mkdir_p(config.checkpoint)

    # Create experiment
    print("==> Creating model")

    exp_module = importlib.import_module("experiments." + args.EXP)
    assert issubclass(exp_module.Experiment, BaseExperiment), args.EXP + ".Experiment is not a subclass of BaseExperiment"
    exp = exp_module.Experiment(hparams)

    if config.resume is None:
        with open(hparams_cp_file, "w") as f:
            YAML().dump(hparams, f)
    else:
        # Load checkpoint
        if args.resume_file is None:
            print("[Error] Please specify resume_file in the argumen")
            sys.exit(1)

        resume_full = os.path.join(config.resume, args.resume_file)
        if not os.path.isfile(resume_full):
            print("[Error] No checkpoint found at '{}'".format(config.resume))
            sys.exit(1)
        
        if not load_checkpoint(exp, config.resume, args.resume_file, args.ignore_hparams_mismatch):
            print("[Error] hparams mismatch or loading failed")
            sys.exit(0)

    config.tb_writer = SummaryWriter(log_dir="runs/{exp_name}_{exp_id}/{datetime}".format(exp_name=exp_name, exp_id=exp.hparams["id"], datetime=datetime.datetime.now().strftime("%b%d_%H-%M-%S")))

    print("==> Initiating dataloader")
    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset,
        collate_fn=exp.train_collate_fn,
        batch_size=exp.hparams["train_batch"],
        num_workers=config.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=exp.train_drop_last,
        worker_init_fn=exp.worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset,
        collate_fn=exp.valid_collate_fn,
        batch_size=exp.hparams["test_batch"],
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=exp.worker_init_fn)

    if config.evaluate:
        print("\n[Info] Evaluation-only mode")
        result_collection = validate(val_loader, exp, 0, 0)
        if result_collection is not None:
            print("")
            save_pred(result_collection, checkpoint=config.checkpoint)
        return

    if config.handle_sig:
        enable_sigint_handler()

    train_eval_loop(exp, exp.hparams["before_epoch"] + 1, exp.hparams["epochs"] + 1, train_loader, val_loader)

def load_checkpoint(exp, checkpoint_folder, checkpoint_file, ignore_hparams_mismatch=False):
    # Checking hparams consistency
    old_hparams_file = os.path.join(checkpoint_folder, "hparams.yaml")
    if os.path.isfile(old_hparams_file):
        with open(old_hparams_file, "r") as f:
            old_hparams = YAML().load(f)
        if not check_hparams_consistency(old_hparams, exp.hparams):
            print("[Warning] hparams from config and from checkpoint are not equal")
            if not ignore_hparams_mismatch:
                print("In current:")
                YAML().dump(exp.hparams, sys.stdout)
                print("<<<<\n>>>>\nIn checkpoint:")
                YAML().dump(old_hparams, sys.stdout)
                if not ask("Continue ?"):
                    return False
            else:
                print("[Info] hparams mismatch ignored")

    # Load checkpoint data
    checkpoint_full = os.path.join(checkpoint_folder, checkpoint_file)
    print("==> Loading checkpoint '{}'".format(checkpoint_full))
    checkpoint = torch.load(checkpoint_full)
    exp.hparams["before_epoch"] = checkpoint["epoch"]
    exp.model.load_state_dict(checkpoint["state_dict"], strict=not args.no_strict_model_load)
    if not args.no_criterion_load:
        exp.criterion.load_state_dict(checkpoint["criterion"])
    if not args.no_optimizer_load:
        exp.optimizer.load_state_dict(checkpoint["optimizer"])
    print("==> Loaded checkpoint (epoch {})".format(checkpoint["epoch"]))

    return True

def train_eval_loop(exp, start_epoch, stop_epoch, train_loader, val_loader):
    for epoch in range(start_epoch, stop_epoch):
        exp.epoch_start(epoch)

        print("\nEpoch: %d | LR: %.8f" % (epoch, exp.cur_lr))

        # train for one epoch
        train(train_loader, exp, epoch, em_valid_int=config.em_valid_int, val_loader=val_loader)

        if config.sigint_triggered:
            return False

        cur_step = len(train_loader) * epoch

        # evaluate on validation set
        if config.skip_val > 0 and epoch % config.skip_val == 0:
            print("Validation:")
            result_collection = validate(val_loader, exp, epoch, cur_step)
        else:
            print("Skip validation")
            result_collection = None

        if config.sigint_triggered:
            return False

        cp_filename = "checkpoint_{}.pth.tar".format(epoch)
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": exp.model.state_dict(),
            "optimizer": exp.optimizer.state_dict(),
            "criterion": exp.criterion.state_dict()
        }
        save_checkpoint(checkpoint_dict, False, checkpoint=config.checkpoint, filename=cp_filename)

        if result_collection is not None:
            preds_filename = "preds_{}.npy".format(epoch)
            save_pred(result_collection, is_best=False, checkpoint=config.checkpoint, filename=preds_filename)

        exp.epoch_end(epoch)

        if config.sigint_triggered:
            return False

    return True

def train(train_loader, exp, epoch, em_valid_int=0, val_loader=None):
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

            cur_step = iter_length * (epoch - 1) + i

            progress = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step
            }

            result = exp.iter_process(epoch_ctx, batch, True, progress=progress)
            loss = result["loss"]

            exp.iter_step(loss)

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
            
            if config.sigint_triggered:
                break

            if em_valid:
                print("\nEmbeded Validation:")
                validate(val_loader, exp, epoch, cur_step + 1, store_result=True)
                print("")
                exp.model.train()
                end = time.time()

            if config.sigint_triggered:
                break

            if config.fast_pass_train > 0 and (i+1) >= config.fast_pass_train:
                print("Fast Pass!")
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
                "step": cur_step
            }

            result = exp.iter_process(epoch_ctx, batch, False, progress=progress)

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

            if config.sigint_triggered:
                break

            if config.fast_pass_valid > 0 and (i+1) >= config.fast_pass_valid:
                print("Fast Pass!")
                break

    exp.summary_scalar_avg(epoch_ctx, cur_step, phase="valid")
    exp.evaluate(preds, cur_step)

    return result_collection

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument("CONF", type=str)
    argp.add_argument("EXP", type=str)
    argp.add_argument("-r", dest="resume_file", type=str)
    argp.add_argument("--no-strict-model-load", action="store_true")
    argp.add_argument("--no-criterion-load", action="store_true")
    argp.add_argument("--no-optimizer-load", action="store_true")
    argp.add_argument("--ptvsd", action="store_true")
    argp.add_argument("--ignore-hparams-mismatch", action="store_true")
    argp.add_argument("--override", nargs=2, metavar=("var", "value"), action="append")
    argp.add_argument("--config", nargs=2, metavar=("var", "value"), action="append")
    return argp.parse_known_args()

def init_config(conf_name, config_override):
    with open("experiments/config.yaml", "r") as f:
        conf = YAML(typ="safe").load(f)
    conf_data = conf[conf_name]
    config.__dict__.update(conf_data.items())
    if config_override:
        config.__dict__.update(dict(map(lambda x: (x[0], eval(str(x[1]))), config_override)))

def get_hparams(exp_name, hp_file="experiments/hparams.yaml"):
    with open(hp_file, "r") as f:
        return YAML().load(f)[exp_name]

if __name__ == "__main__":
    args, unknown_args = get_args()
    if args.ptvsd:
        import ptvsd
        import platform
        ptvsd.enable_attach("mydebug", address = ("0.0.0.0", 23456))
        if platform.node() == "lhtserver-2":
            print("Waiting for debugger...")
            ptvsd.wait_for_attach()
            print("Debugger attached!")

    main(args, unknown_args)
