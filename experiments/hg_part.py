from __future__ import print_function, absolute_import

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

from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import part_accuracy, accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, detect_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
import pose.utils.config as config
import pose.models as models
import pose.datasets as datasets

# Hyperdash
from hyperdash import Experiment
config.hyperdash_exp = None

# Handle sigint
import signal
config.sigint_triggered = False
def sigint_handler(signal, frame):
    config.sigint_triggered = True
    print("SIGINT Detected")

# Profiling
import cProfile, pstats, StringIO
config.profiler = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0

def main(args):
    global best_acc

    config.debug = args.debug

    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    assert args.stacks == 3, "Experiment Requirement"

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks,
                                       num_classes=[15, 15, 16])

    model = torch.nn.DataParallel(model).cuda()

    # loss function and optimizer
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    title = 'mpii-' + args.arch
    logger = None
    logfile = join(args.checkpoint, 'log.txt')
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if isfile(logfile):
                logger = Logger(logfile, title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            sys.exit(1)
    else:
        if isfile(logfile) or detect_checkpoint(checkpoint=args.checkpoint):
            print("Already existed log.txt or checkpoint in %s, using --resume or moving them" % args.checkpoint)
            sys.exit(1)
            
    if not logger:        
        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc'])

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
        datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                      sigma=args.sigma, label_data=datasets.Mpii.LABEL_MIX_MAP,
                      label_type=args.label_type, single_person=True),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                      sigma=args.sigma, label_data=datasets.Mpii.LABEL_MIX_MAP,
                      label_type=args.label_type, train=False, single_person=True),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation only') 
        loss, acc, predictions = validate(val_loader, model, criterion, args.num_classes, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr

    if args.hyperdash:
        config.hyperdash_exp = Experiment("Hourglass-Part")

    if args.profile:
        config.profiler = cProfile.Profile()

    signal.signal(signal.SIGINT, sigint_handler)
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # TODO: sigma_decay not used
        # # decay sigma
        # if args.sigma_decay > 0:
        #     train_loader.dataset.sigma *=  args.sigma_decay
        #     val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        if config.sigint_triggered:
            break

        print("Validation")

        # evaluate on validation set
        valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, args.num_classes, epoch, args.flip)

        if config.sigint_triggered:
            break

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        # remember best acc and save checkpoint
        is_best = valid_acc > best_acc
        best_acc = max(valid_acc, best_acc)

        cp_filename = 'checkpoint_{}.pth.tar'.format(epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename=cp_filename)

        preds_filename = 'preds_{}.npy'.format(epoch + 1)
        save_pred(predictions, is_best=is_best, checkpoint=args.checkpoint, filename=preds_filename)

        if config.sigint_triggered:
            break

    # if config.profiler:
    #     s = StringIO.StringIO()
    #     ps = pstats.Stats(config.profiler, stream=s).sort_stats('cumulative')
    #     ps.print_stats()
    #     print(s.getvalue())

    if config.hyperdash_exp:
        config.hyperdash_exp.end()

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Processing', max=len(train_loader))

    # if config.profiler:
    #     config.profiler.enable()

    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda())
        target_vis_var = torch.autograd.Variable(target['parts_v'].cuda(async=True))
        target_all_var = torch.autograd.Variable(target['parts_a'].cuda(async=True))
        target_pts_var = torch.autograd.Variable(target['points'].cuda(async=True))
        mparts_vis_var = torch.autograd.Variable(meta['mparts_v'].cuda(async=True))
        mparts_all_var = torch.autograd.Variable(meta['mparts_a'].cuda(async=True))

        # compute output
        output = model(input_var)
        score_map = output[-1].data.cpu()

        output_vis = output[0]
        output_all = output[1]
        output_pts = output[2]

        mask_vis = (((target_vis_var < output_vis) + (mparts_vis_var == 2)).eq(2) + (mparts_vis_var == 1)).gt(0)
        mask_all = mparts_all_var

        loss_vis = criterion(output_vis[mask_vis], target_vis_var[mask_vis])
        loss_all = criterion(output_all[mask_all], target_all_var[mask_all])
        loss_pts = criterion(output_pts, target_pts_var)

        loss = loss_vis + loss_all + loss_pts

        # acc = part_accuracy(score_map * mask_all.data.cpu(), target['parts_a'] * mask_all.data.cpu())
        acc = accuracy(score_map, target['points'], idx)

        if config.debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, score_map)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
        #             batch=i + 1,
        #             size=len(train_loader),
        #             data=data_time.val,
        #             bt=batch_time.val,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             acc=acces.avg
        #             )
        # bar.next()
        loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:2.6f}s | Batch: {bt:2.3f}s | Total: {total:} | Loss: {loss:.4f} | Acc: {acc:.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc: .4f}'.format(
                  epoch=epoch + 1,
                  batch=i + 1,
                  size_width=len(str(len(train_loader))),
                  size=len(train_loader),
                  data=data_time.val,
                  bt=batch_time.val,
                  total=bar.elapsed_td,
                  loss=losses.val,
                  acc=acces.val,
                  avgloss=losses.avg,
                  avgacc=acces.avg
                  )
        print(loginfo)

        if config.hyperdash_exp:
            config.hyperdash_exp.metric("accuracy", acces.val, log=False)
            config.hyperdash_exp.metric("loss", losses.val, log=False)
            if i > 10:
                # initial accuracy is inaccurate
                config.hyperdash_exp.metric("avg.accuracy", acces.avg, log=False)
                config.hyperdash_exp.metric("avg.loss", losses.avg, log=False)

        if config.sigint_triggered:
            break

    # if config.profiler: config.profiler.disable()
    # bar.finish()
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, num_classes, epoch, flip=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # predictions
    # predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)
    pred_lim = 50
    predictions_parts = torch.Tensor(pred_lim, 2, 15, 64, 64)
    predictions_pts = torch.Tensor(pred_lim, 16, 64, 64)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(inputs.cuda(), volatile=True)
        target_vis_var = torch.autograd.Variable(target['parts_v'].cuda(async=True), volatile=True)
        target_all_var = torch.autograd.Variable(target['parts_a'].cuda(async=True), volatile=True)
        target_pts_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=True)
        mparts_vis_var = torch.autograd.Variable(meta['mparts_v'].cuda(async=True), volatile=True)
        mparts_all_var = torch.autograd.Variable(meta['mparts_a'].cuda(async=True), volatile=True)

        # compute output
        output = model(input_var)

        output_vis = output[0]
        output_all = output[1]
        output_pts = output[2]

        # mask_vis = ((target_vis_var < output_vis) & (mparts_vis_var == 2)) | (mparts_vis_var == 1)
        mask_vis = (((target_vis_var < output_vis) + (mparts_vis_var == 2)).eq(2) + (mparts_vis_var == 1)).gt(0)
        mask_all = mparts_all_var

        loss_vis = criterion(output_vis[mask_vis], target_vis_var[mask_vis])
        loss_all = criterion(output_all[mask_all], target_all_var[mask_all])
        loss_pts = criterion(output_pts, target_pts_var)

        loss = loss_vis + loss_all + loss_pts

        parts_vis_map = output_vis.data.cpu()
        parts_all_map = output_all.data.cpu()
        points_map = output_pts.data.cpu()

        acc = accuracy(points_map, target['points'], idx)

        if flip:
            flip_input_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr(inputs.clone().numpy())).float().cuda(), 
                    volatile=True
                )
            flip_output_var = model(flip_input_var)
            # flip_output = flip_back(flip_output_var[-1].data.cpu(), datatype='parts')
            flip_output = flip_back(flip_output_var[-1].data.cpu())
            points_map += flip_output
            points_map /= 2.

        # generate predictions
        for n in range(points_map.size(0)):
            if meta['index'][n] >= pred_lim:
                continue
            predictions_parts[meta['index'][n], 0, :, :, :] = parts_vis_map[n, :, :64, :64]
            predictions_parts[meta['index'][n], 1, :, :, :] = parts_all_map[n, :, :64, :64]
            predictions_pts[meta['index'][n], :, :, :] = points_map[n, :, :64, :64]

        if config.debug:
            gt_batch_img = batch_with_heatmap(inputs, target)
            pred_batch_img = batch_with_heatmap(inputs, points_map)
            if not gt_win or not pred_win:
                plt.subplot(121)
                gt_win = plt.imshow(gt_batch_img)
                plt.subplot(122)
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.data[0], inputs.size(0))
        acces.update(acc[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
        #             batch=i + 1,
        #             size=len(val_loader),
        #             data=data_time.val,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             acc=acces.avg
        #             )
        # bar.next()
        
        loginfo = '{epoch:3}: ({batch:0{size_width}}/{size}) Data: {data:2.6f}s | Batch: {bt:2.3f}s | Total: {total:} | Loss: {loss:.4f} | Acc: {acc:.4f} | A.Loss: {avgloss:.4f} | A.Acc: {avgacc: .4f}'.format(
                  epoch=epoch + 1,
                  batch=i + 1,
                  size_width=len(str(len(val_loader))),
                  size=len(val_loader),
                  data=data_time.val,
                  bt=batch_time.avg,
                  total=bar.elapsed_td,
                  loss=losses.val,
                  acc=acces.val,
                  avgloss=losses.avg,
                  avgacc=acces.avg
                  )
        print(loginfo)

        if config.sigint_triggered:
            break

    # bar.finish()
    return losses.avg, acces.avg, {'parts': predictions_parts, 'pts': predictions_pts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=15, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--no-hd', dest='hyperdash', action='store_false',
                        help='disable hyperdash')
    parser.add_argument('--profile', action='store_true',
                        help='profile training')

    # import ipdb; ipdb.set_trace()
    main(parser.parse_args())
