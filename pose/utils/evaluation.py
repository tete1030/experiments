from __future__ import absolute_import

import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.linear_model import LinearRegression
from munkres import Munkres

from .misc import *
from .transforms import transform, transform_preds

__all__ = ['accuracy', 'AverageMeter']

def get_preds(scores):
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:,:,0] = preds[:,:,0] % scores.size(3)
    preds[:,:,1] = torch.floor(preds[:,:,1] / scores.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds = (preds + 1) * pred_mask - 1
    return preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] >= 0 and target[n, c, 1] >= 0:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def accuracy(output, target, idxs, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = get_preds(target)
    norm    = torch.ones(preds.size(0))*output.size(3)/10
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(len(idxs)+1)
    avg_acc = 0
    cnt = 0

    for i in range(len(idxs)):
        acc[i+1] = dist_acc(dists[idxs[i]-1])
        if acc[i+1] >= 0: 
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
    return acc
 
def part_accuracy(output, target, score_thr=0.5, IoU_thr=0.5):
    ''' Calculate the accuracy of body part prediction, similar to IoU'''
    preds = (output > score_thr)
    gts = (target > score_thr)
    union = preds | gts
    its = preds & gts
    
    its_sum = its.sum(3).sum(2)
    union_sum = union.sum(3).sum(2)
    empty_idx = (union_sum == 0)
    its_sum[empty_idx] = 1
    union_sum[empty_idx] = 1
    IoU = its_sum.float() / union_sum.float()
    correct = (IoU > IoU_thr).float()
    chn_acc = correct.mean(0)
    avg_acc = correct.mean()
    
    return torch.cat([torch.FloatTensor([avg_acc]), chn_acc])

def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n, p]
            px = int(math.floor(coords[n, p, 0]))
            py = int(math.floor(coords[n, p, 1]))
            if px >= 0 and px < res[0] and py >= 0 and py < res[1]:
                diff = torch.Tensor([hm[py, px+1] - hm[py, px-1], hm[py+1, px] - hm[py-1, px]])
                coords[n, p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def match_locate(pred_locate, gt, locate_index, threshold_dis=3):
    """Match locate map result with ground truth keypoints
    
    Arguments:
        pred_locate {list of Tensors} -- #batch x [#person x 2]
        gt {list of Tensors} -- #batch x [#person x #part x 3]
        locate_index {int} -- the index of part used for locating
    
    Solution:
        A - for each pred locate point, find ground truth person, ignore remaining locate point
        B - for each ground truth person, find pred locate point, ignore remaining locate point
        C - Hungarian Algorithm
    """

    mk = Munkres()
    matches_pred = []
    matches_gt = []
    for pred_locate_i, gt_i in zip(pred_locate, gt):
        index_gt_labeled = torch.nonzero(gt_i[:, locate_index, 2] > 0)
        if len(index_gt_labeled) == 0 or len(pred_locate_i) == 0:
            matches_pred.append(torch.LongTensor(0))
            matches_gt.append(torch.LongTensor(0))
            continue

        index_gt_labeled = index_gt_labeled[:, 0]
        gt_locate_i = gt_i[index_gt_labeled][:, locate_index, :2].float().contiguous()
        pred_locate_i = pred_locate_i.float()
        
        if gt_locate_i.size(0) > pred_locate_i.size(0):
            rc_order = 'pred_gt'
            locate_max = gt_locate_i
            locate_min = pred_locate_i
            size_max = gt_locate_i.size(0)
            size_min = pred_locate_i.size(0)
        else:
            rc_order = 'gt_pred'
            locate_max = pred_locate_i
            locate_min = gt_locate_i
            size_max = pred_locate_i.size(0)
            size_min = gt_locate_i.size(0)

        locate_min_mat = locate_min.view(-1, 1, 2).repeat(1, size_max, 1)
        locate_max_mat = locate_max.view(1, -1, 2).repeat(size_min, 1, 1)
        diff = ((locate_max_mat - locate_min_mat) ** 2).sum(dim=2)
        diff[diff > threshold_dis**2] = 1e10
        if size_min < size_max:
            diff = torch.cat([diff, torch.Tensor(size_max - size_min, size_max).float().fill_(1e10)], dim=0)

        pairs = mk.compute(diff)
        samp_match_pred = []
        samp_match_gt = []
        for row, col in pairs:
            if row >= size_min or diff[row, col] > threshold_dis**2:
                continue
            if rc_order == 'pred_gt':
                samp_match_pred.append(row)
                samp_match_gt.append(col)
            else:
                samp_match_pred.append(col)
                samp_match_gt.append(row)
        samp_match_pred = torch.LongTensor(samp_match_pred)
        samp_match_gt = index_gt_labeled[torch.LongTensor(samp_match_gt)]
        # Sort by ground truth indecies for convenience
        # samp_match = samp_match[samp_match[:, 1].sort()[1]]
        matches_pred.append(samp_match_pred)
        matches_gt.append(samp_match_gt)
    return matches_pred, matches_gt

def accuracy_locate(locate_pred, gt, locate_index, num_gt, norm, threshold=0.5):
    num_TP = 0
    for ib in range(len(locate_pred)):
        pred_i = locate_pred[ib]
        gt_i = gt[ib]
        if len(gt_i) > 0:
            mask_labeled = (gt_i[:, locate_index, 2] > 0)
            pred_i = pred_i.float()
            gt_i = gt_i[:, locate_index, :2]
            mask_TP = ((((pred_i - gt_i) ** 2).sum(dim=-1) / (float(norm) ** 2)) <= threshold ** 2)
            mask_TP = mask_TP & mask_labeled
            num_TP += mask_TP.sum()
    return float(num_TP) / float(num_gt)

def accuracy_multi(pred, gt, norm, num_parts, threshold=0.5):
    """Calculate accuracy of multi-person predictions
    
    Arguments:
        pred {list of Tensors} -- #batch x [#batch_i_person x #part x 3]
        gt_all {list of Tensors} -- #batch x [#batch_i_person x #part x 3]
        norm {int} -- normalize distance to 1
        num_parts {int} -- number of parts
    
    Keyword Arguments:
        threshold {float} -- threshold for distance (default: {0.5})
    """
    
    counter_all = torch.zeros(num_parts).float()
    counter_TP = torch.zeros(num_parts).float()

    for ib in range(len(pred)):
        pred_i = pred[ib]
        gt_i = gt[ib]
        if len(gt_i) > 0:
            mask_labeled = (gt_i[:, :, 2] > 0)
            pred_i = pred_i[:, :, :2].float()
            gt_i = gt_i[:, :, :2]
            mask_TP = ((((pred_i - gt_i) ** 2).sum(dim=-1) / (float(norm) ** 2)) <= threshold ** 2)
            mask_TP = mask_TP & mask_labeled
            counter_TP += mask_TP.float().sum(dim=0)
            counter_all += mask_labeled.float().sum(dim=0)

    accs = (counter_TP / counter_all)
    # TODO: more elegant way to process NAN number show in final
    mask_labeled_all = (counter_all > 0)
    # TODO: temporary assert
    assert mask_labeled_all.sum() > 0
    final = accs[mask_labeled_all].mean()
    return final, accs
