from __future__ import print_function, absolute_import
import torch
import pose.models as models
import pose.datasets as datasets
from pose.utils.transforms import fliplr_chwimg, fliplr_map
from pose.utils.evaluation import accuracy
from utils.globals import config, hparams, globalvars
from utils.train import adjust_learning_rate
import cv2
import numpy as np
import torchvision.utils as vutils

INP_RES=256
OUT_RES=64

class Experiment(object):
    def __init__(self):

        self.model = torch.nn.DataParallel(
            models.HourglassNet(models.Bottleneck,
                num_stacks=hparams['model']['stack'],
                num_blocks=hparams['model']['block'],
                num_classes=datasets.mpii.NUM_PARTS)).cuda()
    
        self.criterion = torch.nn.MSELoss(size_average=True).cuda()
        
        self.optimizer = torch.optim.RMSprop(list(self.model.parameters()),
                                             lr=hparams['learning_rate'],
                                             weight_decay=hparams['weight_decay'])

        self.train_dataset = datasets.MPII(
            img_folder='data/mpii/images',
            anno_file='data/mpii/mpii_human_pose.json',
            split_file='data/mpii/split_sig.pth',
            meanstd_file='data/mpii/mean_std.pth',
            train=True,
            single_person=True,
            inp_res=INP_RES,
            out_res=OUT_RES,
            label_sigma=1)
    
        self.val_dataset = datasets.MPII(
            img_folder='data/mpii/images',
            anno_file='data/mpii/mpii_human_pose.json',
            split_file='data/mpii/split_sig.pth',
            meanstd_file='data/mpii/mean_std.pth',
            train=False,
            single_person=True,
            inp_res=INP_RES,
            out_res=OUT_RES,
            label_sigma=1)

        self.train_collate_fn = datasets.MPII.collate_function
        self.test_collate_fn = datasets.MPII.collate_function
    
    def epoch(self, epoch):
        hparams['learning_rate'] = adjust_learning_rate(self.optimizer, epoch, hparams['learning_rate'], hparams['schedule'], hparams['lr_gamma'])
        # decay sigma
        label_sigma_decay = hparams['dataset']['label_sigma_decay']
        if label_sigma_decay > 0:
            self.train_dataset.label_sigma *= label_sigma_decay
            self.val_dataset.label_sigma *= label_sigma_decay

    def summary_image(self, img, pred, gt, title, step):
        # FIXME: not finished
        tb_num = 3
        tb_img = img[:tb_num].numpy() + self.train_dataset.mean[None, :, None, None]
        tb_gt = gt[:tb_num].numpy()
        tb_pred = pred[:tb_num].numpy()
        show_img = np.zeros((tb_num * 2, 3, INP_RES, INP_RES))
        for iimg in range(tb_num):
            cur_img = (tb_img[iimg][::-1].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            cur_gt = cv2.applyColorMap(
                cv2.resize(
                    (tb_gt[iimg].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                    (INP_RES, INP_RES)),
                cv2.COLORMAP_HOT)
            cur_gt = cv2.addWeighted(cur_img, 1, cur_gt, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
            show_img[iimg] = cur_gt
            cur_pred = cv2.applyColorMap(
                cv2.resize(
                    (tb_pred[iimg].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8),
                    (INP_RES, INP_RES)),
                cv2.COLORMAP_HOT)
            cur_pred = cv2.addWeighted(cur_img, 1, cur_pred, 0.5, 0).transpose(2, 0, 1)[::-1].astype(np.float32) / 255
            show_img[tb_num + iimg] = cur_pred

        show_img = vutils.make_grid(torch.from_numpy(show_img), nrow=tb_num, range=(0, 1))
        globalvars.tb_writer.add_image(title, show_img, step)

    def summary_histogram(self, n_iter):
        for name, param in self.model.named_parameters():
            globalvars.tb_writer.add_histogram("hg." + name, param.clone().cpu().data.numpy(), n_iter, bins="doane")

    def process(self, batch, train, detail=None):
        img, target, extra = batch
        volatile = not train
        img_var = torch.autograd.Variable(img.cuda(), volatile=volatile)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=volatile)
    
        output = self.model(img_var)
        score_map = output[-1].data.cpu()
    
        loss = self.criterion(output[0], target_var)
        for j in range(1, len(output)):
            loss += self.criterion(output[j], target_var)
    
        if not train and hparams['model']['flip']:
            flip_img_var = torch.autograd.Variable(
                    torch.from_numpy(fliplr_chwimg(img.numpy())).float().cuda(), 
                    volatile=True
                )
            flip_output_var = self.model(flip_img_var)
            flip_output = torch.from_numpy(
                fliplr_map(flip_output_var[-1].data.cpu().numpy(), datasets.mpii.FLIP_INDEX))
            score_map += flip_output
            score_map /= 2.
    
        acc = accuracy(score_map, target, datasets.mpii.EVAL_INDEX)

        if ("summary" in detail and detail["summary"]):
            self.summary_image(img, score_map.cpu(), target, "hg/" + ("train" if train else "val"), detail["epoch"] + 1)

        result = {
            "loss": loss,
            "acc": acc[0],
            "recall": acc[0],
            "prec": None,
            "index": extra["index"],
            "pred": score_map
        }
    
        return result
