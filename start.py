from __future__ import print_function, absolute_import
import matplotlib
matplotlib.use("TkAgg")

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

# Hyperdash
import hyperdash

# Handle sigint
import signal
config.sigint_triggered = False
def enable_sigint_handler():
    def sigint_handler(signal, frame):
        config.sigint_triggered = True
        print("SIGINT Detected")
    signal.signal(signal.SIGINT, sigint_handler)

best_acc = 0

def main(args):
    global best_acc

    exp_name = args.EXP

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
                # TODO: FIXME
                resume_hparams = YAML().load(open(hparams_cp_file, 'r'))
                assert resume_hparams == exp.hparams, "hparams from config and from checkpoint are not equal"
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
        YAML().dump(exp.hparams, open(hparams_cp_file, 'w'))

    if logger is None:
        logger = Logger(log_file, title=title)
        # TODO: STRUCTURE dynamic metrics
        # logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])
        logger.set_names(['Epoch', 'LR',
                          'Train Locate Loss', 'Train Pose Loss',
                          'Val Locate Loss', 'Val Pose Loss',
                          'Train Locate Acc', 'Train Pose Acc',
                          'Val Locate Acc', 'Val Pose Acc'])

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset, collate_fn=exp.train_collate_fn \
                                          if 'train_collate_fn' in exp.__dict__ else default_collate,
        batch_size=exp.hparams['train_batch'], shuffle=True, 
        num_workers=config.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset, collate_fn=exp.test_collate_fn \
                                        if 'test_collate_fn' in exp.__dict__ else default_collate,
        batch_size=exp.hparams['test_batch'], shuffle=False,
        num_workers=config.workers, pin_memory=True)

    if config.evaluate:
        print('\nEvaluation-only mode')
        loss, acc, predictions = validate(val_loader, exp, 0)
        save_pred(predictions, checkpoint=config.checkpoint)
        return

    if config.hyperdash:
        config.hyperdash = hyperdash.Experiment(config.hyperdash)

    if config.handle_sig:
        enable_sigint_handler()

    for epoch in range(exp.hparams['start_epoch'], exp.hparams['epochs']):
        exp.epoch(epoch)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, exp.hparams['learning_rate']))

        # train for one epoch
        # TODO: STRUCTURE dynamic metrics
        train_loss, train_loss_locate, train_loss_pose, train_acc_locate, train_acc_pose = train(train_loader, exp, epoch)

        if config.sigint_triggered:
            break

        # evaluate on validation set
        if config.skip_val > 0 and (epoch + 1) % config.skip_val == 0:
            print("Validation:")
            # TODO: STRUCTURE dynamic metrics
            valid_loss, valid_loss_locate, valid_loss_pose, valid_acc_locate, valid_acc_pose, predictions = validate(val_loader, exp, epoch)
            # remember best acc and save checkpoint
            is_best = valid_acc_pose > best_acc
            best_acc = max(valid_acc_pose, best_acc)
        else:
            print("Skip validation")
            valid_loss, valid_loss_locate, valid_loss_pose, valid_acc_locate, valid_acc_pose, predictions = 0., 0., 0., 0., 0., None
            is_best = False

        if config.sigint_triggered:
            break

        # append logger file
        # TODO: STRUCTURE dynamic metrics
        logger.append([epoch + 1, exp.hparams['learning_rate'],
                       train_loss_locate, train_loss_pose,
                       valid_loss_locate, valid_loss_pose,
                       train_acc_locate, train_acc_pose,
                       valid_acc_locate, valid_acc_pose])

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

    if config.hyperdash:
        config.hyperdash.end()

    logger.close()
    # TODO: STRUCTURE dynamic metrics
    logger.plot(['Train Locate Loss', 'Train Pose Loss',
                 'Val Locate Loss', 'Val Pose Loss',
                 'Train Locate Acc', 'Train Pose Acc',
                 'Val Locate Acc', 'Val Pose Acc'])
    savefig(os.path.join(config.checkpoint, 'log.eps'))

def train(train_loader, exp, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # acces = AverageMeter()
    # TODO: STRUCTURE dynamic metrics
    losses_locate = AverageMeter()
    acces_locate = AverageMeter()
    losses_pose = AverageMeter()
    acces_pose = AverageMeter()

    # switch to train mode
    exp.model.train()

    end = time.time()
    start_time = time.time()

    gt_win, pred_win = None, None

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO: STRUCTURE dynamic metrics
        # loss, loss_locate, loss_pose, acc_locate, acc_pose, extras["index"], keypoint_pred_match
        loss, loss_locate, loss_pose, acc_locate, acc_pose, _, _ = exp.process(batch, True)

        exp.optimizer.zero_grad()
        loss.backward()
        exp.optimizer.step()

        batch_size = batch[0].size(0)
        # measure accuracy and record loss
        losses.update(loss.data[0], batch_size)
        # TODO: STRUCTURE dynamic metrics
        losses_locate.update(loss_locate.data[0], batch_size)
        if loss_pose is not None:
            losses_pose.update(loss_pose.data[0], batch_size)
        if acc_locate is not None:
            acces_locate.update(acc_locate, batch_size)
        if acc_pose is not None:
            acces_pose.update(acc_pose, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO: STRUCTURE dynamic metrics
        # loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s | Loss: {loss:.4f} | Acc: {acc:7.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc:7.4f}'.format(
        #           epoch=epoch + 1,
        #           batch=i + 1,
        #           size_width=len(str(len(train_loader))),
        #           size=len(train_loader),
        #           data=data_time.val,
        #           bt=batch_time.val,
        #           total=time.time() - start_time,
        #           loss=losses.val,
        #           acc=acces.val,
        #           avgloss=losses.avg,
        #           avgacc=acces.avg
        #           )
        loginfo = ('{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s\n' + \
                       '\tLoss: {loss:.4f} | LossL: {loss_locate:.4f} | LossP: {loss_pose:.4f} | AccL: {acc_locate:7.4f} | AccP: {acc_pose:7.4f}\n' + \
                       '\tLos_: {loss_avg:.4f} | Los_L: {loss_locate_avg:.4f} | Los_P: {loss_pose_avg:.4f} | Ac_L: {acc_locate_avg:7.4f} | Ac_P: {acc_pose_avg:7.4f}').format(
            epoch=epoch + 1,
            batch=i + 1,
            size_width=len(str(len(train_loader))),
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=time.time() - start_time,
            loss=losses.val,
            loss_locate=losses_locate.val,
            # may not be updated
            loss_pose=loss_pose.data[0] if loss_pose is not None else -1.,
            acc_locate=acc_locate if acc_locate is not None else -1.,
            acc_pose=acc_pose if acc_pose is not None else -1.,
            loss_avg=losses.avg,
            loss_locate_avg=losses_locate.avg,
            loss_pose_avg=losses_pose.avg,
            acc_locate_avg=acces_locate.avg,
            acc_pose_avg=acces_pose.avg
        )
        print(loginfo)

        if config.hyperdash:
            config.hyperdash.metric("acc", acces.val, log=False)
            config.hyperdash.metric("loss", losses.val, log=False)
            if i > 10:
                # initial accuracy is inaccurate
                config.hyperdash.metric("acc_avg", acces.avg, log=False)
                config.hyperdash.metric("loss_avg", losses.avg, log=False)

        if config.sigint_triggered:
            break

        if config.fast_pass > 0 and (i+1) >= config.fast_pass:
            print("Fast Pass!")
            break
    
    # TODO: STRUCTURE dynamic metrics
    return losses.avg, losses_locate.avg, losses_pose.avg, acces_locate.avg, acces_pose.avg

def validate(val_loader, exp, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # acces = AverageMeter()
    # TODO: STRUCTURE dynamic metrics
    losses_locate = AverageMeter()
    acces_locate = AverageMeter()
    losses_pose = AverageMeter()
    acces_pose = AverageMeter()

    predictions = None

    # switch to evaluate mode
    exp.model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    start_time = time.time()
    data_counter = 0
    for i, batch in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # loss, acc, index, pred = exp.process(batch, False)
        # TODO: STRUCTURE dynamic metrics
        loss, loss_locate, loss_pose, acc_locate, acc_pose, index, pred = exp.process(batch, False)

        if index is None:
            index = list(range(data_counter, data_counter+len(index)))

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

        batch_size = batch[0].size(0)
        # measure accuracy and record loss
        losses.update(loss.data[0], batch[0].size(0))
        # acces.update(acc, batch[0].size(0))
        # TODO: STRUCTURE dynamic metrics
        losses_locate.update(loss_locate.data[0], batch_size)
        if loss_pose is not None:
            losses_pose.update(loss_pose.data[0], batch_size)
        if acc_locate is not None:
            acces_locate.update(acc_locate, batch_size)
        if acc_pose is not None:
            acces_pose.update(acc_pose, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        data_counter += len(index)

        if config.hyperdash:
            config.hyperdash.metric("acc_val", acces.val, log=False)
            config.hyperdash.metric("loss_val", losses.val, log=False)
            if i > 10:
                # initial accuracy is inaccurate
                config.hyperdash.metric("acc_val_avg", acces.avg, log=False)
                config.hyperdash.metric("loss_val_avg", losses.avg, log=False)
        
        # loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s | Loss: {loss:.4f} | Acc: {acc:7.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc:7.4f}'.format(
        #           epoch=epoch + 1,
        #           batch=i + 1,
        #           size_width=len(str(len(val_loader))),
        #           size=len(val_loader),
        #           data=data_time.val,
        #           bt=batch_time.avg,
        #           total=time.time() - start_time,
        #           loss=losses.val,
        #           acc=acces.val,
        #           avgloss=losses.avg,
        #           avgacc=acces.avg
        #           )
        # TODO: STRUCTURE dynamic metrics
        loginfo = ('{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s\n' + \
                       '\tLoss: {loss:.4f} | LossL: {loss_locate:.4f} | LossP: {loss_pose:.4f} | AccL: {acc_locate:7.4f} | AccP: {acc_pose:7.4f}\n' + \
                       '\tLos_: {loss_avg:.4f} | Los_L: {loss_locate_avg:.4f} | Los_P: {loss_pose_avg:.4f} | Ac_L: {acc_locate_avg:7.4f} | Ac_P: {acc_pose_avg:7.4f}').format(
            epoch=epoch + 1,
            batch=i + 1,
            size_width=len(str(len(val_loader))),
            size=len(val_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=time.time() - start_time,
            loss=losses.val,
            loss_locate=losses_locate.val,
            # may not be updated
            loss_pose=loss_pose.data[0] if loss_pose is not None else -1.,
            acc_locate=acc_locate if acc_locate is not None else -1.,
            acc_pose=acc_pose if acc_pose is not None else -1.,
            loss_avg=losses.avg,
            loss_locate_avg=losses_locate.avg,
            loss_pose_avg=losses_pose.avg,
            acc_locate_avg=acces_locate.avg,
            acc_pose_avg=acces_pose.avg
        )
        print(loginfo)

        if config.sigint_triggered:
            break

        if config.fast_pass > 0 and (i+1) >= config.fast_pass:
            print("Fast Pass!")
            break

    # TODO: STRUCTURE dynamic metrics
    return losses.avg, losses_locate.avg, losses_pose.avg, acces_locate.avg, acces_pose.avg, predictions

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('CONF', type=str)
    argp.add_argument('EXP', type=str)
    argp.add_argument('-r', dest='resume_file', type=str, default='model_best.pth.tar')
    argp.add_argument('--ptvsd', action='store_true')
    return argp.parse_args()

def init_config(conf_name):
    conf = YAML().load(open('experiments/config.yaml', 'r'))
    conf_data = conf[conf_name]
    config.__dict__.update(conf_data.items())
    
def get_hparams(exp_name, hp_file='experiments/hparams.yaml'):
    return YAML().load(open(hp_file, 'r'))[exp_name]

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
