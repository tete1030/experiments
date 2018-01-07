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
from pose.utils.evaluation import part_accuracy, accuracy, AverageMeter, final_preds
from pose.utils.misc import save_checkpoint, detect_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr_chwimg, fliplr_map
import pose.utils.config as config
import pose.models as models
import pose.datasets as datasets
import numpy as np

# Hyperdash
from hyperdash import Experiment
config.hyperdash_exp = None

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

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

PART_VISIBLE_NC = 15
PART_ALL_NC = 15
POINT_NC = 16

idx = [1,2,3,4,5,6,11,12,15,16]

best_acc = 0

def main(args, train_loader=None, val_loader=None):
    global best_acc

    config.debug = args.debug
    config.exp = args.exp
    config.fastpass = args.fastpass

    assert config.exp in ['part', 'ori', 'inf', 'path', 'weight']

    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    if config.exp == 'part':
        assert args.stacks == 3, "Experiment Requirement"
        model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks,
                                           num_classes=[PART_VISIBLE_NC, PART_ALL_NC, POINT_NC])
    elif config.exp == 'ori':
        model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks,
                                           num_classes=POINT_NC)
    elif config.exp == 'inf':
        model = models.MaskHourglassNet(models.Bottleneck, 
                                        num_stacks=args.stacks,
                                        num_blocks=args.blocks,
                                        num_classes=POINT_NC,
                                        mask=True)
    elif config.exp == 'path':
        model = models.PathHourglassNet(models.Bottleneck,
                                         num_stacks=args.stacks,
                                         num_blocks=args.blocks,
                                         num_classes=POINT_NC)

    elif config.exp == 'weight':
        model = models.HourglassNet(models.Bottleneck,
                                    num_stacks=args.stacks,
                                    num_blocks=args.blocks,
                                    num_classes=POINT_NC)

    model = torch.nn.DataParallel(model).cuda()

    # loss function and optimizer
    if config.exp != 'weight':
        criterion = torch.nn.MSELoss(size_average=True).cuda()
    else:
        init_weight = np.zeros((POINT_NC, args.stacks), dtype=np.float32)
        init_weight[[7, 8, 9, 12, 13], 0] = 1.
        init_weight[[7, 8, 9, 12, 13, 6, 11, 14, 2, 3], 1] = 1.
        init_weight[[7, 8, 9, 12, 13, 6, 11, 14, 2, 3, 10, 15, 1, 4, 0, 5], 2] = 1.
        init_weight = torch.from_numpy(init_weight)
        criterion = models.HGDynamicWeightMSELoss(args.stacks, POINT_NC, init_weight=init_weight, dynamic=False).cuda()

    params = list(model.parameters())
    if config.exp == 'weight':
        params += filter(lambda p: p.requires_grad, criterion.parameters())
    optimizer = torch.optim.RMSprop(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    title = 'mpii-' + config.exp + '-' + args.arch
    logger = None
    logfile = join(args.checkpoint, 'log.txt')
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            if config.exp == 'weight':
                criterion.load_state_dict(checkpoint['criterion_SD'])
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

    selective = None
    if args.selective:
        selective_batch_count = 200
        mpii_train_size = 22246
        if isfile(args.selective):
            selective = np.load(args.selective)
        else:
            print("Selective not exist, generating ...")
            selective = np.arange(mpii_train_size, dtype=int)
            np.random.shuffle(selective)
            selective = selective[:selective_batch_count * args.train_batch]
            np.save(args.selective, selective)

    if config.exp == 'part':
        label_data = datasets.Mpii.LABEL_MIX_MAP
        is_single_person = True
    elif config.exp in ['ori', 'inf', 'path', 'weight']:
        label_data = datasets.Mpii.LABEL_POINTS_MAP
        is_single_person = True

    if train_loader is None:
        # Data loading code
        train_loader = torch.utils.data.DataLoader(
            datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                          sigma_pts=args.sigma_pts, label_data=label_data,
                          label_type=args.label_type, single_person=is_single_person,
                          selective=selective, train=True, contrast_factor=0.,
                          brightness_factor=0.),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    
    if val_loader is None:
        val_loader = torch.utils.data.DataLoader(
            datasets.Mpii('data/mpii/mpii_annotations.json', 'data/mpii/images',
                          sigma_pts=args.sigma_pts, label_data=label_data,
                          label_type=args.label_type, train=False, single_person=is_single_person),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        print('\nEvaluation-only mode') 
        loss, acc, predictions = validate(val_loader, model, criterion, args.flip)
        save_pred(predictions, checkpoint=args.checkpoint)
        return

    lr = args.lr

    if args.hyperdash:
        config.hyperdash_exp = Experiment(args.hyperdash)

    if args.profile:
        assert args.workers == 0, "Profiling doesn't support multi-processing"
        config.profiler = cProfile.Profile()

    if args.handle_sig:
        enable_sigint_handler()

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma_pts *=  args.sigma_decay
            val_loader.dataset.sigma_pts *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        #if config.exp == 'weight':
        #    print(criterion.state_dict()['weight'])

        if config.sigint_triggered:
            break

        # evaluate on validation set
        if args.skip_val > 0 and (epoch + 1) % args.skip_val == 0:
            print("Validation:")
            valid_loss, valid_acc, predictions = validate(val_loader, model, criterion, epoch, args.flip)
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
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc])

        cp_filename = 'checkpoint_{}.pth.tar'.format(epoch + 1)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }
        if config.exp == 'weight':
            checkpoint_dict['criterion_SD'] = criterion.state_dict()
        save_checkpoint(checkpoint_dict, is_best, checkpoint=args.checkpoint, filename=cp_filename)

        if predictions is not None:
            preds_filename = 'preds_{}.npy'.format(epoch + 1)
            save_pred(predictions, is_best=is_best, checkpoint=args.checkpoint, filename=preds_filename)

        if config.sigint_triggered:
            break

    if config.profiler:
        s = StringIO.StringIO()
        ps = pstats.Stats(config.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        print(s.getvalue())

    if config.hyperdash_exp:
        config.hyperdash_exp.end()

    logger.close()
    logger.plot(['Train Acc', 'Val Acc'])
    savefig(os.path.join(args.checkpoint, 'log.eps'))

def process_part(model, criterion, optimizer, batch, train, flip=False):
    inputs, target, meta = batch
    volatile = not train
    input_var = torch.autograd.Variable(inputs.cuda(), volatile=volatile)
    target_vis_var = torch.autograd.Variable(target['parts_v'].cuda(async=True), volatile=volatile)
    target_all_var = torch.autograd.Variable(target['parts_a'].cuda(async=True), volatile=volatile)
    target_pts_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=volatile)
    mparts_vis_var = torch.autograd.Variable(meta['mparts_v'].cuda(async=True), volatile=volatile)
    mparts_all_var = torch.autograd.Variable(meta['mparts_a'].cuda(async=True), volatile=volatile)

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

    if train:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not train and flip:
        flip_input_var = torch.autograd.Variable(
                torch.from_numpy(fliplr_chwimg(inputs.numpy())).float().cuda(), 
                volatile=True
            )
        flip_output_var = model(flip_input_var)
        # flip_output = fliplr_map(flip_output_var[-1].data.cpu(), datatype='parts')
        flip_output = fliplr_map(flip_output_var[-1].data.cpu())
        points_map += flip_output
        points_map /= 2.

    acc = accuracy(points_map, target['points'], idx)

    if config.debug: # visualize groundtruth and predictions
        gt_batch_img = batch_with_heatmap(inputs, target['points'])
        pred_batch_img = batch_with_heatmap(inputs, points_map)
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

    if train:
        return loss.data[0], acc[0]
    else:
        return loss.data[0], acc[0], (parts_vis_map, parts_all_map, points_map)

def process_inf(model, criterion, optimizer, batch, train, flip=False):
    inputs, target, meta = batch
    volatile = not train
    input_var = torch.autograd.Variable(inputs.cuda(), volatile=volatile)
    target_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=volatile)
    point_num = target_var.size(1)

    output = model(input_var)

    map_acc = 0.
    mask_acc = 0.
    map_loss = 0.
    mask_loss = 0.
    map_level_diff = 0.

    for j in range(0, len(output)):
        map_med = output[j][:, :point_num]
        mask_med = output[j][:, point_num:]
        map_acc = map_acc + map_med * mask_med
        mask_acc = mask_acc + mask_med
        map_level_diff = map_level_diff + (map_med - target_var).abs() * mask_med
    
    # map_loss = criterion(map_acc, target_var)
    map_loss = (map_level_diff ** 2).sum() / map_level_diff.numel() + criterion(map_acc, target_var)
    # mask_loss = mask_acc.sub(1.).pow(2).sum() / mask_acc.numel()

    loss = map_loss

    if train:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    score_map = map_acc.data.cpu()
    if not train and flip:
        flip_input_var = torch.autograd.Variable(
                torch.from_numpy(fliplr_chwimg(inputs.numpy())).float().cuda(), 
                volatile=True
            )
        flip_output_var = model(flip_input_var)

        flip_map_acc = 0.
        for j in range(0, len(output)):
            flip_map_med = flip_output_var[j][:, :point_num]
            flip_mask_med = flip_output_var[j][:, point_num:]
            flip_map_acc = flip_map_med * flip_mask_med + flip_map_acc
            
        flip_output = fliplr_map(flip_map_acc.data.cpu())
        score_map += flip_output
        score_map /= 2.

    acc = accuracy(score_map, target['points'], idx)

    if config.debug: # visualize groundtruth and predictions
        gt_batch_img = batch_with_heatmap(inputs, target['points'])
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

    if train:
        return loss.data[0], acc[0]
    else:
        return loss.data[0], acc[0], score_map

def process_weight(model, criterion, optimizer, batch, train, flip=False):
    inputs, target, meta = batch
    volatile = not train
    input_var = torch.autograd.Variable(inputs.cuda(), volatile=volatile)
    target_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=volatile)

    output = model(input_var)
    score_map = output[-1].data.cpu()

    loss = criterion(output, target_var)

    if train:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not train and flip:
        flip_input_var = torch.autograd.Variable(
                torch.from_numpy(fliplr_chwimg(inputs.numpy())).float().cuda(), 
                volatile=True
            )
        flip_output_var = model(flip_input_var)
        # flip_output = fliplr_map(flip_output_var[-1].data.cpu(), datatype='parts')
        flip_output = fliplr_map(flip_output_var[-1].data.cpu())
        score_map += flip_output
        score_map /= 2.

    acc = accuracy(score_map, target['points'], idx)

    if config.debug: # visualize groundtruth and predictions
        gt_batch_img = batch_with_heatmap(inputs, target['points'])
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

    if train:
        return loss.data[0], acc[0]
    else:
        return loss.data[0], acc[0], score_map

def process_ori(model, criterion, optimizer, batch, train, flip=False):
    inputs, target, meta = batch
    volatile = not train
    input_var = torch.autograd.Variable(inputs.cuda(), volatile=volatile)
    target_var = torch.autograd.Variable(target['points'].cuda(async=True), volatile=volatile)

    output = model(input_var)
    score_map = output[-1].data.cpu()

    loss = criterion(output[0], target_var)
    for j in range(1, len(output)):
        loss += criterion(output[j], target_var)

    if train:
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not train and flip:
        flip_input_var = torch.autograd.Variable(
                torch.from_numpy(fliplr_chwimg(inputs.numpy())).float().cuda(), 
                volatile=True
            )
        flip_output_var = model(flip_input_var)
        # flip_output = fliplr_map(flip_output_var[-1].data.cpu(), datatype='parts')
        flip_output = fliplr_map(flip_output_var[-1].data.cpu())
        score_map += flip_output
        score_map /= 2.

    acc = accuracy(score_map, target['points'], idx)

    if config.debug: # visualize groundtruth and predictions
        gt_batch_img = batch_with_heatmap(inputs, target['points'])
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

    if train:
        return loss.data[0], acc[0]
    else:
        return loss.data[0], acc[0], score_map

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    start_time = time.time()

    gt_win, pred_win = None, None

    if config.profiler: config.profiler.enable()

    for i, (inputs, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.exp == 'part':
            loss, acc = process_part(model, criterion, optimizer, (inputs, target, meta), train=True)
        elif config.exp == 'ori':
            loss, acc = process_ori(model, criterion, optimizer, (inputs, target, meta), train=True)
        elif config.exp == 'inf':
            loss, acc = process_inf(model, criterion, optimizer, (inputs, target, meta), train=True)
        elif config.exp == 'path':
            # TODO ori
            loss, acc = process_ori(model, criterion, optimizer, (inputs, target, meta), train=True)
        elif config.exp == 'weight':
            loss, acc = process_weight(model, criterion, optimizer, (inputs, target, meta), train=True)

        # measure accuracy and record loss
        losses.update(loss, inputs.size(0))
        acces.update(acc, inputs.size(0))

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

        if config.hyperdash_exp:
            config.hyperdash_exp.metric("acc", acces.val, log=False)
            config.hyperdash_exp.metric("loss", losses.val, log=False)
            if i > 10:
                # initial accuracy is inaccurate
                config.hyperdash_exp.metric("acc_avg", acces.avg, log=False)
                config.hyperdash_exp.metric("loss_avg", losses.avg, log=False)

        if config.sigint_triggered:
            break

        if config.fastpass > 0 and (i+1) >= config.fastpass:
            print("Fast Pass!")
            break

    if config.profiler: config.profiler.disable()
    return losses.avg, acces.avg

def validate(val_loader, model, criterion, epoch, flip=False, pred_lim=50):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    pred_res = 64
    # predictions
    if config.exp == 'part':
        predictions_parts = torch.Tensor(pred_lim, PART_VISIBLE_NC + PART_ALL_NC, pred_res, pred_res)
        predictions_pts = torch.Tensor(pred_lim, POINT_NC, pred_res, pred_res)
        predictions = {'parts': predictions_parts, 'pts': predictions_pts}
    elif config.exp in ['ori', 'inf', 'path', 'weight']:
        # predictions = torch.Tensor(val_loader.dataset.__len__(), POINT_NC, 2)
        predictions = torch.Tensor(pred_lim, POINT_NC, pred_res, pred_res)

    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    start_time = time.time()
    for i, (inputs, target, meta) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.exp == 'part':
            loss, acc, (parts_vis_map, parts_all_map, points_map) = \
                    process_part(model, criterion, None, (inputs, target, meta), train=False, flip=flip)

            # generate predictions
            for n in range(points_map.size(0)):
                if meta['index'][n] >= pred_lim:
                    continue
                predictions_parts[meta['index'][n], :PART_VISIBLE_NC, :, :] = parts_vis_map[n, :, :pred_res, :pred_res]
                predictions_parts[meta['index'][n], PART_VISIBLE_NC:, :, :] = parts_all_map[n, :, :pred_res, :pred_res]
                predictions_pts[meta['index'][n], :, :, :] = points_map[n, :, :pred_res, :pred_res]

        elif config.exp in ['ori', 'inf', 'path', 'weight']:
            # TODO ori
            process_func = {'ori': process_ori, 'inf': process_inf, 'path': process_ori, 'weight': process_weight}[config.exp]
            loss, acc, score_map = process_func(model, criterion, None, (inputs, target, meta), train=False, flip=flip)

            for n in range(score_map.size(0)):
                if meta['index'][n] >= pred_lim:
                    continue
                # predictions[meta['index'][n], :, :] = preds[n, :, :]
                predictions[meta['index'][n], :, :, :] = score_map[n, :, :pred_res, :pred_res]

        # measure accuracy and record loss
        losses.update(loss, inputs.size(0))
        acces.update(acc, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if config.hyperdash_exp:
            config.hyperdash_exp.metric("acc_val", acces.val, log=False)
            config.hyperdash_exp.metric("loss_val", losses.val, log=False)
            if i > 10:
                # initial accuracy is inaccurate
                config.hyperdash_exp.metric("acc_val_avg", acces.avg, log=False)
                config.hyperdash_exp.metric("loss_val_avg", losses.avg, log=False)
        
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

        if config.fastpass > 0 and (i+1) >= config.fastpass:
            print("Fast Pass!")
            break

    return losses.avg, acces.avg, predictions

def get_parser():
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
    parser.add_argument('--sigma-pts', type=float, default=1,
                        help='Groundtruth Gaussian sigma for points.')
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
    parser.add_argument('--profile', action='store_true',
                        help='profile training')
    parser.add_argument('--selective', type=str, metavar='FILE',
                        help='select a part of dataset')
    parser.add_argument('--hyperdash', type=str, metavar='HDNAME',
                        help='name used by hyperdash')
    parser.add_argument('--exp', type=str, metavar='EXPNAME',
                        help='experiment name')
    parser.add_argument('--skip-val', type=int, metavar='EPOCH_COUNT',
                        default=1, help='do validation for every EPOCH_COUNT epochs (default: 1)')
    parser.add_argument('--fastpass', nargs='?', type=int, const=2,
                        default=0, help='for fast test')
    parser.add_argument('--no-handle-sig', dest='handle_sig', action='store_false',
                        help='do not handle SIGINT')
    return parser

if __name__ == '__main__':
    main(get_parser().parse_args())
