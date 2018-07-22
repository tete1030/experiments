#!python3
import os
import sys
import argparse
import time
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

import collections
from tensorboardX import SummaryWriter
import datetime
from experiments.baseexperiment import BaseExperiment, EpochContext

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

# Handle sigint
import signal
import multiprocessing
config.sigint_triggered = False
def enable_sigint_handler():
    ori_sigint_handler = signal.getsignal(signal.SIGINT)
    def sigint_handler(signal, frame):
        if config.sigint_triggered:
            ori_sigint_handler(signal, frame)
        config.sigint_triggered = True
        if type(multiprocessing.current_process()) != multiprocessing.Process:
            print("SIGINT")
    signal.signal(signal.SIGINT, sigint_handler)

best_acc = 0

def main(args):
    global best_acc

    exp_name = args.EXP
    config.exp_name = exp_name

    hparams = get_hparams(exp_name)

    if args.override is not None:
        def set_hierarchic_attr(var, var_name_hierarchic, var_value):
            if len(var_name_hierarchic) > 1:
                set_hierarchic_attr(var[var_name_hierarchic[0]], var_name_hierarchic[1:], var_value)
            else:
                var[var_name_hierarchic[0]] = var_value
        for var_name, var_value in args.override:
            set_hierarchic_attr(hparams, var_name.split("."), eval(var_value))
        set_hierarchic_attr = None

    config.checkpoint = config.checkpoint.format(**{'exp': exp_name, 'id': hparams['id']})
    if config.resume is not None:
        config.resume = config.resume.format(**{'exp': exp_name, 'id': hparams['id']})

    hparams_cp_file = os.path.join(config.checkpoint, 'hparams.yaml')
    if not config.resume and ( \
            detect_checkpoint(checkpoint=config.checkpoint) or \
            os.path.isfile(hparams_cp_file)):
        print("Exist files in %s" % config.checkpoint)
        ans_del = input("Do you want to delete files in %s (yes|n): " % config.checkpoint)
        if ans_del not in ["yes", "n"]:
            print("Wrong answer. Exit.")
            sys.exit(1)
        if ans_del == "yes":
            print("Deleting %s" % config.checkpoint)
            import shutil
            shutil.rmtree(config.checkpoint)
        else:
            print("Not delete. Exit.")
            sys.exit(0)

    print("==> creating model")

    if not os.path.isdir(config.checkpoint):
        mkdir_p(config.checkpoint)

    exp_module = importlib.import_module('experiments.' + hparams['name'])
    assert issubclass(exp_module.Experiment, BaseExperiment)
    exp = exp_module.Experiment(hparams)
    del hparams

    if config.resume:
        resume_full = os.path.join(config.resume, args.resume_file)
        if os.path.isfile(resume_full):
            hparams_cp_file = os.path.join(config.resume, 'hparams.yaml')
            if os.path.isfile(hparams_cp_file):
                with open(hparams_cp_file, 'r') as f:
                    resume_hparams = YAML(typ='safe').load(f)
                if resume_hparams != exp.hparams:
                    print("Warning: hparams from config and from checkpoint are not equal")
                    if not args.ignore_hparams_differ:
                        print("In config:")
                        YAML(typ='safe').dump(exp.hparams, sys.stdout)
                        print("<<<<\n>>>>\nIn checkpoint:")
                        YAML(typ='safe').dump(resume_hparams, sys.stdout)
                        ans = input("Continue (y|n)? ")
                        if ans != "y":
                            sys.exit(0)
                    else:
                        print("Difference ignored")
            print("=> loading checkpoint '{}'".format(resume_full))
            checkpoint = torch.load(resume_full)
            exp.hparams['start_epoch'] = checkpoint['epoch']
            exp.model.load_state_dict(checkpoint['state_dict'], strict=not args.no_strict_model_load)
            if not args.no_criterion_load:
                exp.criterion.load_state_dict(checkpoint['criterion'])
            if not args.no_optimizer_load:
                exp.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))
            sys.exit(1)
    else:
        with open(hparams_cp_file, 'w') as f:
            YAML(typ='safe').dump(exp.hparams, f)

    config.tb_writer = SummaryWriter(log_dir="runs/{exp_name}_{exp_id}/{datetime}".format(exp_name=exp_name, exp_id=exp.hparams["id"], datetime=datetime.datetime.now().strftime("%b%d_%H-%M-%S")))

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset,
        collate_fn=exp.train_collate_fn,
        batch_size=exp.hparams['train_batch'],
        num_workers=config.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=exp.train_drop_last,
        worker_init_fn=exp.worker_init_fn)

    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset,
        collate_fn=exp.valid_collate_fn,
        batch_size=exp.hparams['test_batch'],
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=exp.worker_init_fn)

    if config.evaluate:
        print('\nEvaluation-only mode')
        result_collection = validate(val_loader, exp, 0, 0)
        if result_collection is not None:
            save_pred(result_collection, checkpoint=config.checkpoint)
        return

    if config.handle_sig:
        enable_sigint_handler()

    for epoch in range(exp.hparams['start_epoch'], exp.hparams['epochs']):
        exp.epoch_start(epoch)

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, exp.cur_lr))

        # train for one epoch
        train(train_loader, exp, epoch, em_valid_int=exp.hparams["em_valid_int"], val_loader=val_loader)

        if config.sigint_triggered:
            break

        cur_step = len(train_loader) * (epoch + 1)

        # evaluate on validation set
        if config.skip_val > 0 and (epoch + 1) % config.skip_val == 0:
            print("Validation:")
            result_collection = validate(val_loader, exp, epoch, cur_step)
        else:
            print("Skip validation")
            result_collection = None

        if config.sigint_triggered:
            break

        cp_filename = 'checkpoint_{}.pth.tar'.format(epoch + 1)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'state_dict': exp.model.state_dict(),
            'optimizer': exp.optimizer.state_dict(),
            'criterion': exp.criterion.state_dict()
        }
        save_checkpoint(checkpoint_dict, False, checkpoint=config.checkpoint, filename=cp_filename)

        if result_collection is not None:
            preds_filename = 'preds_{}.npy'.format(epoch + 1)
            save_pred(result_collection, is_best=False, checkpoint=config.checkpoint, filename=preds_filename)

        exp.epoch_end(epoch)

        if config.sigint_triggered:
            break

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

            cur_step = iter_length * epoch + i

            detail = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step
            }

            result = exp.iter_process(epoch_ctx, batch, True, detail=detail)
            loss = result["loss"]

            exp.iter_step(loss)

            loss = loss.item() if loss is not None else None

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            loginfo = ("{epoch:3}: ({batch:0{size_width}}/{size}) data: {data:.6f}s | batch: {bt:.3f}s | total: {total:3.1f}s").format(
                epoch=epoch + 1,
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

            if config.fast_pass > 0 and (i+1) >= config.fast_pass:
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

            detail = {
                "epoch": epoch,
                "iter": i,
                "iter_len": iter_length,
                "step": cur_step
            }

            result = exp.iter_process(epoch_ctx, batch, False, detail=detail)

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
                epoch=epoch + 1,
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

            if config.fast_pass > 0 and (i+1) >= config.fast_pass:
                print("Fast Pass!")
                break

    exp.summary_scalar_avg(epoch_ctx, cur_step, phase="valid")
    exp.evaluate(preds)

    return result_collection

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('CONF', type=str)
    argp.add_argument('EXP', type=str)
    argp.add_argument('-r', dest='resume_file', type=str, default='model_best.pth.tar')
    argp.add_argument('--no-strict-model-load', action='store_true')
    argp.add_argument('--no-criterion-load', action='store_true')
    argp.add_argument('--no-optimizer-load', action='store_true')
    argp.add_argument('--ptvsd', action='store_true')
    argp.add_argument('--ignore-hparams-differ', action='store_true')
    argp.add_argument('--override', nargs=2, metavar=('var', 'value'), action='append')
    argp.add_argument('--config', nargs=2, metavar=('var', 'value'), action='append')
    return argp.parse_args()

def init_config(conf_name, config_override):
    with open('experiments/config.yaml', 'r') as f:
        conf = YAML(typ='safe').load(f)
    conf_data = conf[conf_name]
    config.__dict__.update(conf_data.items())
    if config_override:
        config.__dict__.update(dict(map(lambda x: (x[0], eval(str(x[1]))), config_override)))

def get_hparams(exp_name, hp_file='experiments/hparams.yaml'):
    with open(hp_file, 'r') as f:
        return YAML(typ='safe').load(f)[exp_name]

if __name__ == '__main__':
    args = get_args()
    if args.ptvsd:
        import ptvsd
        import platform
        ptvsd.enable_attach("mydebug", address = ('0.0.0.0', 23456))
        if platform.node() == 'lhtserver-2':
            print("Waiting for debugger...")
            ptvsd.wait_for_attach()
            print("Debugger attached!")

    init_config(args.CONF, args.config)
    main(args)
