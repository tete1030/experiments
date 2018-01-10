from __future__ import print_function, absolute_import
import matplotlib
matplotlib.use('Agg')

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

# Profiling
import cProfile, pstats, StringIO
config.profiler = None

best_acc = 0

def main(args):
    global best_acc

    exp_name = args.EXP

    hparams = get_hparams(exp_name)
    config.checkpoint = config.checkpoint.format(**{'exp': exp_name, 'id': hparams['id']})
    if config.resume is not None:
        config.resume = config.resume.format(**{'exp': exp_name, 'id': hparams['id']})

    print("==> creating model")

    if not os.path.isdir(config.checkpoint):
        mkdir_p(config.checkpoint)

    exp_module = importlib.import_module('experiments.' + hparams['name'])
    exp = exp_module.Experiment(hparams)
    del hparams

    title = exp_name
    logger = None
    log_file = os.path.join(config.checkpoint, 'log.txt')
    if config.resume:
        resume_full = os.path.join(config.resume, args.resume_file)
        if os.path.isfile(resume_full):
            hparams_cp_file = os.path.join(config.resume, 'hparams.yaml')
            if os.path.isfile(hparams_cp_file):
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
        hparams_cp_file = os.path.join(config.checkpoint, 'hparams.yaml')
        YAML().dump(exp.hparams, open(hparams_cp_file, 'w'))
        if os.path.isfile(log_file) or \
           detect_checkpoint(checkpoint=config.checkpoint) or \
           os.path.isfile(hparams_cp_file):
            print("Exist files in %s, resume or delete them" % config.checkpoint)
            sys.exit(1)
            
    if logger is None: 
        logger = Logger(log_file, title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        exp.train_dataset,
        batch_size=exp.hparams['train_batch'], shuffle=True,
        num_workers=config.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        exp.val_dataset,
        batch_size=exp.hparams['test_batch'], shuffle=False,
        num_workers=config.workers, pin_memory=True)

    if config.evaluate:
        print('\nEvaluation-only mode') 
        loss, acc, predictions = validate(val_loader, exp, 0)
        save_pred(predictions, checkpoint=config.checkpoint)
        return

    if config.hyperdash:
        config.hyperdash = hyperdash.Experiment(config.hyperdash)

    if config.profile:
        assert config.workers == 0, "Profiling doesn't support multi-processing"
        config.profiler = cProfile.Profile()

    if config.handle_sig:
        enable_sigint_handler()

    for epoch in range(exp.hparams['start_epoch'], exp.hparams['epochs']):
        exp.epoch(epoch)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, exp.hparams['learning_rate']))

        # train for one epoch
        train_loss, train_acc = train(train_loader, exp, epoch)

        if config.sigint_triggered:
            break

        # evaluate on validation set
        if config.skip_val > 0 and (epoch + 1) % config.skip_val == 0:
            print("Validation:")
            valid_loss, valid_acc, predictions = validate(val_loader, exp, epoch)
            # remember best acc and save checkpoint
            is_best = valid_acc > best_acc
            best_acc = max(valid_acc, best_acc)
        else:
            print("Skip validation")
            valid_loss, valid_acc, predictions = 0., 0., None
            is_best = False

        if config.sigint_triggered:
            break

        # append logger file
        logger.append([epoch + 1, exp.hparams['learning_rate'], train_loss, valid_loss, train_acc, valid_acc])

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

    if config.profiler:
        s = StringIO.StringIO()
        ps = pstats.Stats(config.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

    if config.hyperdash:
        config.hyperdash.end()

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(config.checkpoint, 'log.eps'))

def train(train_loader, exp, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    exp.model.train()

    end = time.time()
    start_time = time.time()

    gt_win, pred_win = None, None

    if config.profiler: config.profiler.enable()

    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # TODO
        loss, acc, _, _ = exp.process(batch, True)

        exp.optimizer.zero_grad()
        loss.backward()
        exp.optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.data[0], batch[0].size(0))
        acces.update(acc, batch[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s | Loss: {loss:.4f} | Acc: {acc:7.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc:7.4f}'.format(
                  epoch=epoch + 1,
                  batch=i + 1,
                  size_width=len(str(len(train_loader))),
                  size=len(train_loader),
                  data=data_time.val,
                  bt=batch_time.val,
                  total=time.time() - start_time,
                  loss=losses.val,
                  acc=acces.val,
                  avgloss=losses.avg,
                  avgacc=acces.avg
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

    if config.profiler: config.profiler.disable()
    return losses.avg, acces.avg

def validate(val_loader, exp, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

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

        loss, acc, index, score_map = exp.process(batch, False)

        if index is None:
            index = list(range(data_counter, data_counter+len(index)))

        if predictions is None:
            predictions = torch.zeros((config.pred_lim,) + score_map.size()[1:])

        for n in range(score_map.size(0)):
            if index[n] >= config.pred_lim:
                continue
            predictions[index[n]] = score_map[n]

        # measure accuracy and record loss
        losses.update(loss.data[0], batch[0].size(0))
        acces.update(acc, batch[0].size(0))

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
        
        loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:3.1f}s | Loss: {loss:.4f} | Acc: {acc:7.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc:7.4f}'.format(
                  epoch=epoch + 1,
                  batch=i + 1,
                  size_width=len(str(len(val_loader))),
                  size=len(val_loader),
                  data=data_time.val,
                  bt=batch_time.avg,
                  total=time.time() - start_time,
                  loss=losses.val,
                  acc=acces.val,
                  avgloss=losses.avg,
                  avgacc=acces.avg
                  )
        print(loginfo)

        if config.sigint_triggered:
            break

        if config.fast_pass > 0 and (i+1) >= config.fast_pass:
            print("Fast Pass!")
            break

    return losses.avg, acces.avg, predictions

def get_args():
    argp = argparse.ArgumentParser()
    argp.add_argument('CONF', type=str)
    argp.add_argument('EXP', type=str)
    argp.add_argument('-r', dest='resume_file', type=str, default='model_best.pth.tar')
    return argp.parse_args()

def init_config(conf_name):
    conf = YAML().load(open('experiments/config.yaml', 'r'))
    conf_data = conf[conf_name]
    config.__dict__.update(conf_data.items())
    
def get_hparams(exp_name, hp_file='experiments/hparams.yaml'):
    return YAML().load(open(hp_file, 'r'))[exp_name]

if __name__ == '__main__':
    args = get_args()
    init_config(args.CONF)
    main(args)
