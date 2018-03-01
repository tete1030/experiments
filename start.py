from __future__ import print_function, absolute_import

# Handle matplotlib backend error when DISPLAY is wrong
# the error originates from Tk used in matplotlib
matplotlib_backend = "TkAgg"
try:
    import Tkinter
    Tkinter.Tk().destroy()
except Tkinter.TclError:
    print("Cannot use TkAgg for matplotlib, using Agg")
    matplotlib_backend = "Agg"
finally:
    del Tkinter
import matplotlib
matplotlib.use(matplotlib_backend)

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
from torch.utils.data.dataloader import default_collate

from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import AverageMeter
from pose.utils.misc import save_checkpoint, detect_checkpoint, save_pred
from pose.utils.osutils import mkdir_p
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr_chwimg, fliplr_map
import pose.utils.config as config
import pose.models as models
import pose.datasets as datasets

import numpy as np
import importlib
from ruamel.yaml import YAML

import collections
from tensorboardX import SummaryWriter

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
    config.checkpoint = config.checkpoint.format(**{'exp': exp_name, 'id': hparams['id']})
    if config.resume is not None:
        config.resume = config.resume.format(**{'exp': exp_name, 'id': hparams['id']})

    title = exp_name
    logger = None
    log_file = os.path.join(config.checkpoint, 'log.txt')
    hparams_cp_file = os.path.join(config.checkpoint, 'hparams.yaml')
    if not config.resume and (os.path.isfile(log_file) or \
            detect_checkpoint(checkpoint=config.checkpoint) or \
            os.path.isfile(hparams_cp_file)):
        print("Exist files in %s" % config.checkpoint)
        ans_del = raw_input("Do you want to delete files in %s (yes|n): " % config.checkpoint)
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
                    print("In config:")
                    YAML(typ='safe').dump(exp.hparams, sys.stdout)
                    print("In checkpoint:")
                    YAML(typ='safe').dump(resume_hparams, sys.stdout)
                    ans = raw_input("Continue (y|n)? ")
                    if ans != "y":
                        sys.exit(0)
            print("=> loading checkpoint '{}'".format(resume_full))
            checkpoint = torch.load(resume_full)
            exp.hparams['start_epoch'] = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            exp.model.load_state_dict(checkpoint['state_dict'])
            exp.criterion.load_state_dict(checkpoint['criterion'])
            exp.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
            if os.path.isfile(log_file):
                logger = Logger(log_file, title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))
            sys.exit(1)
    else:
        with open(hparams_cp_file, 'w') as f:
            YAML(typ='safe').dump(exp.hparams, f)

    config.tb_writer = SummaryWriter()

    if logger is None:
        logger = Logger(log_file, title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Train Prec', 'Val Prec'])

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset,
        collate_fn=exp.train_collate_fn if 'train_collate_fn' in exp.__dict__ else default_collate,
        batch_size=exp.hparams['train_batch'],
        num_workers=config.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=exp.train_drop_last if 'train_drop_last' in exp.__dict__ else False)

    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset,
        collate_fn=exp.test_collate_fn if 'test_collate_fn' in exp.__dict__ else default_collate,
        batch_size=exp.hparams['test_batch'],
        num_workers=config.workers,
        shuffle=False,
        pin_memory=True)

    if config.evaluate:
        print('\nEvaluation-only mode')
        loss, acc, prec, predictions = validate(val_loader, exp, 0)
        save_pred(predictions, checkpoint=config.checkpoint)
        return

    if config.handle_sig:
        enable_sigint_handler()

    for epoch in range(exp.hparams['start_epoch'], exp.hparams['epochs']):
        exp.epoch(epoch)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, exp.hparams['learning_rate']))

        # train for one epoch
        train_loss, train_acc, train_prec = train(train_loader, exp, epoch, em_valid_int=exp.hparams["em_valid_int"], val_loader=val_loader)

        if config.sigint_triggered:
            break

        exp.summary_histogram(len(train_loader) * (epoch + 1))

        # evaluate on validation set
        if config.skip_val > 0 and (epoch + 1) % config.skip_val == 0:
            print("Validation:")
            valid_loss, valid_acc, valid_prec, predictions = validate(val_loader, exp, epoch)
            config.tb_writer.add_scalars(config.exp_name + "/loss", {"valid": valid_loss}, len(train_loader) * (epoch + 1))
            config.tb_writer.add_scalars(config.exp_name + "/acc", {"valid": valid_acc}, len(train_loader) * (epoch + 1))
            config.tb_writer.add_scalars(config.exp_name + "/prec", {"valid": valid_prec}, len(train_loader) * (epoch + 1))
            # remember best acc and save checkpoint
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
        else:
            print("Skip validation")
            valid_loss, valid_acc, valid_prec, predictions = 0., 0., None
            is_best = False

        if config.sigint_triggered:
            break

        # append logger file
        logger.append([epoch + 1, exp.hparams['learning_rate'],
                       train_loss, valid_loss,
                       train_acc, valid_acc,
                       train_prec, valid_prec])

        cp_filename = 'checkpoint_{}.pth.tar'.format(epoch + 1)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'state_dict': exp.model.state_dict(),
            'best_acc': best_acc,
            'optimizer': exp.optimizer.state_dict(),
            'criterion': exp.criterion.state_dict()
        }
        save_checkpoint(checkpoint_dict, is_best, checkpoint=config.checkpoint, filename=cp_filename)

        if predictions is not None:
            preds_filename = 'preds_{}.npy'.format(epoch + 1)
            save_pred(predictions, is_best=is_best, checkpoint=config.checkpoint, filename=preds_filename)

        if config.sigint_triggered:
            break

    logger.close()
    logger.plot(['Train Loss', 'Val Loss',
                 'Train Acc', 'Val Acc',
                 'Train Prec', 'Val Prec'])
    savefig(os.path.join(config.checkpoint, 'log.eps'))

def train(train_loader, exp, epoch, em_valid_int=0, val_loader=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    preces = AverageMeter()

    # switch to train mode
    exp.model.train()

    end = time.time()
    start_time = time.time()

    iter_length = len(train_loader)

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        detail = {
            "epoch": epoch,
            "iter": i,
            "iter_len": iter_length,
            "summary": False
        }
        if (i == iter_length - 1) or (config.fast_pass > 0 and i == config.fast_pass - 1):
            detail["summary"] = True

        result = exp.process(batch, True, detail=detail)
        loss = result["loss"]
        acc = result["acc"]
        prec = result["prec"]

        exp.optimizer.zero_grad()
        loss.backward()
        exp.optimizer.step()

        batch_size = batch[0].size(0)
        loss = loss.data[0] if loss is not None else None
        # measure accuracy and record loss
        if loss is not None:
            losses.update(loss, batch_size)
            config.tb_writer.add_scalars(config.exp_name + "/loss", {"train": loss}, iter_length * epoch + i)
        if acc is not None:
            acces.update(acc, batch_size)
            config.tb_writer.add_scalars(config.exp_name + "/acc", {"train": acc}, iter_length * epoch + i)
        if prec is not None:
            preces.update(prec, batch_size)
            config.tb_writer.add_scalars(config.exp_name + "/prec", {"train": prec}, iter_length * epoch + i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loginfo = ("{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s\n" +
                   "\tLoss: {loss:.4f} | Acc: {acc:7.4f} | Prec: {prec:7.4f}\n" +
                   "\tLos_: {avgloss:.4f} | Ac_: {avgacc:7.4f} | Pre_: {avgprec:7.4f}").format(
            epoch=epoch + 1,
            batch=i + 1,
            size_width=len(str(iter_length)),
            size=iter_length,
            data=data_time.val,
            bt=batch_time.val,
            total=time.time() - start_time,
            loss=loss if loss is not None else -1.,
            acc=acc if acc is not None else -1.,
            prec=prec if prec is not None else -1.,
            avgloss=losses.avg,
            avgacc=acces.avg,
            avgprec=preces.avg
        )
        print(loginfo)

        if config.sigint_triggered:
            break
        
        if em_valid_int > 0 and (i+1) % em_valid_int == 0 and iter_length - (i+1) >= max(em_valid_int/2, 1):
            print("\nEmbeded Validation:")
            valid_loss, valid_acc, valid_prec, predictions = validate(val_loader, exp, epoch, store_pred=False)
            config.tb_writer.add_scalars(config.exp_name + "/loss", {"valid": valid_loss}, iter_length * epoch + i + 1)
            config.tb_writer.add_scalars(config.exp_name + "/acc", {"valid": valid_acc}, iter_length * epoch + i + 1)
            config.tb_writer.add_scalars(config.exp_name + "/prec", {"valid": valid_prec}, iter_length * epoch + i + 1)
            # exp.summary_histogram(iter_length * epoch + i + 1)
            print("")
            exp.model.train()
            end = time.time()
        
        if config.sigint_triggered:
            break

        if config.fast_pass > 0 and (i+1) >= config.fast_pass:
            print("Fast Pass!")
            break

    return losses.avg, acces.avg, preces.avg

def validate(val_loader, exp, epoch, store_pred=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    preces = AverageMeter()

    predictions = None

    # switch to evaluate mode
    exp.model.eval()

    end = time.time()
    start_time = time.time()
    iter_length = len(val_loader)
    data_counter = 0
    for i, batch in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = batch[0].size(0)

        detail = {
            "epoch": epoch,
            "iter": i,
            "iter_len": iter_length,
            "summary": False
        }
        if i == 0:
            detail["summary"] = store_pred

        result = exp.process(batch, False, detail=detail)
        loss = result["loss"]
        loss = loss.data[0] if loss is not None else None
        acc = result["acc"]
        prec = result["prec"]
        index = result["index"] if "index" in result else None
        pred = result["pred"] if "pred" in result else None

        if index is None:
            index = list(range(data_counter, data_counter+batch_size))

        if pred is not None and store_pred:
            if predictions is None:
                if isinstance(pred, torch._TensorBase):
                    predictions = torch.zeros((config.pred_lim,) + pred.size()[1:])
                elif isinstance(pred, collections.Sequence):
                    predictions = dict()
                else:
                    raise TypeError("Not valid pred type")

            for n in range(len(pred)):
                if index[n] >= config.pred_lim:
                    continue
                predictions[index[n]] = pred[n]

        # measure accuracy and record loss
        if loss is not None:
            losses.update(loss, batch_size)
        if acc is not None:
            acces.update(acc, batch_size)
        if prec is not None:
            preces.update(prec, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        data_counter += len(index)
        
        loginfo = ("{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s\n" + 
                   "\tLoss: {loss:.4f} | Acc: {acc:7.4f} | Prec: {prec:7.4f}\n" +
                   "\tLos_: {avgloss:.4f} | Ac_: {avgacc:7.4f} | Pre_: {avgprec:7.4f}").format(
            epoch=epoch + 1,
            batch=i + 1,
            size_width=len(str(iter_length)),
            size=iter_length,
            data=data_time.val,
            bt=batch_time.val,
            total=time.time() - start_time,
            loss=loss if loss is not None else -1.,
            acc=acc if acc is not None else -1.,
            prec=prec if prec is not None else -1.,
            avgloss=losses.avg,
            avgacc=acces.avg,
            avgprec=preces.avg
        )
        print(loginfo)

        if config.sigint_triggered:
            break

        if config.fast_pass > 0 and (i+1) >= config.fast_pass:
            print("Fast Pass!")
            break

    return losses.avg, acces.avg, preces.avg, predictions

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('CONF', type=str)
    argp.add_argument('EXP', type=str)
    argp.add_argument('-r', dest='resume_file', type=str, default='model_best.pth.tar')
    argp.add_argument('--ptvsd', action='store_true')
    return argp.parse_args()

def init_config(conf_name):
    with open('experiments/config.yaml', 'r') as f:
        conf = YAML(typ='safe').load(f)
    conf_data = conf[conf_name]
    config.__dict__.update(conf_data.items())
    
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

    init_config(args.CONF)
    main(args)
