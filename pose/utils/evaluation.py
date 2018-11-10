import math
import queue
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.linear_model import LinearRegression
import torch

from .transforms import transform, transform_preds
from utils.softnms import cpu_soft_nms
import munkres
import copy

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

class CycleAverageMeter(object):
    """Computes and stores the cycled average and current value"""
    def __init__(self, size):
        self.size = size
        self._pool = list()
        self._pointer = 0
        self.reset()

    def reset(self):
        self._pool.clear()
        self._pointer = 0
        self.count = 0
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        for i in range(n):
            if self.count >= self.size:
                self._pool[self._pointer] = val
                self._pointer = (self._pointer + 1) % self.size
            else:
                self._pool.append(val)
                self.count += 1

        self.val = val
        self.avg = float(sum(self._pool)) / self.count

def match_locate(pred, gt, threshold_abandon=3):
    """Match locate map result with ground truth keypoints
    
    Arguments:
        pred {list of Tensors} -- #batch x [#person x 2]
        gt {list of Tensors} -- #batch x [#person x 2]
        threshold_abandon {float} -- threshold distance of abandon
    
    Solution:
        A - for each pred locate point, find ground truth person, ignore remaining locate point
        B - for each ground truth person, find pred locate point, ignore remaining locate point
        C - Hungarian Algorithm
    """

    mk = munkres.Munkres()
    matches_pred = []
    matches_gt = []
    for pred_i, gt_i in zip(pred, gt):
        if len(gt_i) == 0 or len(pred_i) == 0:
            matches_pred.append(torch.LongTensor(0))
            matches_gt.append(torch.LongTensor(0))
            continue

        gt_i = gt_i.float()
        pred_i = pred_i.float()
        
        if gt_i.size(0) > pred_i.size(0):
            rc_order = 'pred_gt'
            locate_max = gt_i
            locate_min = pred_i
            size_max = gt_i.size(0)
            size_min = pred_i.size(0)
        else:
            rc_order = 'gt_pred'
            locate_max = pred_i
            locate_min = gt_i
            size_max = pred_i.size(0)
            size_min = gt_i.size(0)

        locate_min_mat = locate_min.view(-1, 1, 2).repeat(1, size_max, 1)
        locate_max_mat = locate_max.view(1, -1, 2).repeat(size_min, 1, 1)
        diff = ((locate_max_mat - locate_min_mat) ** 2).sum(dim=2)
        diff[diff > threshold_abandon**2] = 1e10
        if size_min < size_max:
            diff = torch.cat([diff, torch.Tensor(size_max - size_min, size_max).float().fill_(1e10)], dim=0)
        diff = diff.tolist()

        pairs = mk.compute(diff)
        samp_match_pred = []
        samp_match_gt = []
        for row, col in pairs:
            if diff[row][col] > threshold_abandon**2:
                continue
            if rc_order == 'pred_gt':
                samp_match_pred.append(row)
                samp_match_gt.append(col)
            else:
                samp_match_pred.append(col)
                samp_match_gt.append(row)
        samp_match_pred = torch.LongTensor(samp_match_pred)
        samp_match_gt = torch.LongTensor(samp_match_gt)
        # Sort by ground truth indecies for convenience
        # samp_match = samp_match[samp_match[:, 1].sort()[1]]
        matches_pred.append(samp_match_pred)
        matches_gt.append(samp_match_gt)
    return matches_pred, matches_gt

def PR_locate(pred, gt, match_pred, match_gt, threshold):
    counter_GT = 0
    counter_P = 0
    counter_TP = 0
    indices_TP = list()
    for ib in range(len(pred)):
        pred_i = pred[ib]
        gt_i = gt[ib]
        counter_P += len(pred_i)
        counter_GT += len(gt_i)
        match_pred_i = match_pred[ib]
        match_gt_i = match_gt[ib]
        assert len(match_pred_i) == len(match_gt_i)
        if len(match_pred_i) > 0:
            pred_i = pred_i[match_pred_i].float()
            gt_i = gt_i[match_gt_i].float()
            mask_TP = (((pred_i - gt_i) ** 2).sum(dim=-1) <= float(threshold) ** 2)
            index_TP_match = mask_TP.nonzero()
            if len(index_TP_match) > 0:
                indices_TP.append(match_pred_i[index_TP_match[:, 0]])
            else:
                indices_TP.append(torch.LongTensor(0))
            counter_TP += mask_TP.int().sum()
        else:
            indices_TP.append(torch.LongTensor(0))

    precision = float(counter_TP) / float(counter_P) if counter_P > 0 else None
    recall = float(counter_TP) / float(counter_GT) if counter_GT > 0 else None
    return precision, recall, indices_TP

def PR_multi(pred, gt, person_norm, num_parts, threshold):
    """Calculate accuracy of multi-person predictions

    Arguments:
        pred {list of Tensors} -- #batch x [#batch_i_person x #part x 3]
        gt {list of Tensors} -- #batch x [#batch_i_person x #part x 3]
        person_norm {list of Tensors} -- #batch x [#batch_i_person]
        num_parts {int} -- number of parts
        threshold {float} -- threshold for distance
    """

    counter_GT = torch.zeros(num_parts).float()
    counter_TP = torch.zeros(num_parts).float()
    counter_P = torch.zeros(num_parts).float()
    indices_TP = list()

    for ib in range(len(pred)):
        pred_i = pred[ib]
        gt_i = gt[ib]
        assert len(gt_i) == len(pred_i)
        if len(gt_i) > 0:
            mask_GT = (gt_i[:, :, 2] > 0)
            mask_P = (pred_i[:, :, 2] > 0)
            pred_i = pred_i[:, :, :2].float()
            gt_i = gt_i[:, :, :2].float()
            mask_thres = ((((pred_i - gt_i) ** 2).sum(dim=-1)).sqrt_() / (person_norm[ib].view(-1, 1).expand(-1, num_parts) if person_norm is not None else 1.) <= float(threshold))
            mask_TP = mask_GT & mask_P & mask_thres
            index_TP = mask_TP.nonzero()
            if len(index_TP) > 0:
                indices_TP.append([(idx_pair[0], idx_pair[1]) for idx_pair in index_TP])
            else:
                indices_TP.append([])
            counter_TP += mask_TP.float().sum(dim=0)
            counter_P += mask_P.float().sum(dim=0)
            counter_GT += mask_GT.float().sum(dim=0)
        else:
            indices_TP.append((torch.LongTensor(0), torch.LongTensor(0)))

    precision = (counter_TP / counter_P)
    recall = (counter_TP / counter_GT)
    final_precision = counter_TP.sum() / counter_P.sum()
    final_recall = counter_TP.sum() / counter_GT.sum()
    return final_precision, final_recall, precision, recall, indices_TP

def generate_person_det_ans(image_ids, rois, roi_scores):
    assert len(image_ids) == len(rois) and len(rois) == len(roi_scores)
    anns = []
    for i in range(len(image_ids)):
        anns.append({
            "image_id": image_ids[i],
            "bbox": [rois[i, 0], rois[i, 1], rois[i, 2]-rois[i, 0], rois[i, 3]-rois[i, 1]],
            "score": roi_scores[i]
        })
    return anns

def filter_person_det(det_anns, nms=False, min_score=1e-10, min_box_size=0, return_ind=False, softnms_sigma=0.5, softnms_thres=0.001):
    last_img_id = -1
    last_img_id_start = -1
    num_anns = len(det_anns)
    ann_ind = 0
    all_ind_keep = list()
    all_dets = list()
    while True:
        if ann_ind >= num_anns or det_anns[ann_ind]["image_id"] != last_img_id:
            if last_img_id_start >= 0:
                # DO
                num_dets = ann_ind - last_img_id_start
                dets = np.zeros((num_dets, 5), dtype=np.float32)
                for i in range(num_dets):
                    bbox = det_anns[last_img_id_start+i]["bbox"]
                    dets[i, :4] = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    dets[i, 4] = det_anns[last_img_id_start+i]["score"]

                ind_keep = np.nonzero(
                    (dets[:, 4] >= min_score) &
                    ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]) >= min_box_size))[0]
                dets = np.ascontiguousarray(dets[ind_keep])

                if nms:
                    ind_keep_nms = cpu_soft_nms(dets, sigma=softnms_sigma, threshold=softnms_thres, method=2)
                    dets = dets[:ind_keep_nms.shape[0]]
                    ind_keep = ind_keep[ind_keep_nms]

                ind_keep += last_img_id_start
                all_dets.append(dets)
                all_ind_keep.append(ind_keep)

            last_img_id_start = ann_ind

        if ann_ind >= num_anns:
            break
        last_img_id = det_anns[ann_ind]["image_id"]
        ann_ind += 1

    all_dets = np.concatenate(all_dets, axis=0)
    all_ind_keep = np.concatenate(all_ind_keep, axis=0)
    assert len(all_ind_keep) == len(all_dets)
    if return_ind:
        return all_ind_keep, all_dets

    det_anns = copy.deepcopy(np.array(det_anns)[all_ind_keep].tolist())
    for ind in range(len(det_anns)):
        det_anns[ind]["score"] = all_dets[ind, 4]
    return det_anns

def nms_oks(kp_predictions, rois, scores, thresh):
    """Nms based on kp predictions."""
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def softnms_oks(kp_predictions, rois, scores, sigma=0.5, thresh=0.001):
    """Nms based on kp predictions."""
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i],
            kp_predictions[order[1:]], rois[order[1:]])
        weight = np.exp(- (ovr)**2 / sigma)
        scores[order[1:]] *= weight
        inds = np.where(weight > thresh)[0]
        order = order[inds + 1]

    return keep

def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: Kx3
    src_roi: 4
    dst_keypoints: NxKx3
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    dx = dst_keypoints[:, :, 0] - src_keypoints[:, 0]
    dy = dst_keypoints[:, :, 1] - src_keypoints[:, 1]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e

def keypoints_nms(anns, rois, num_parts, return_ind=False, method="nms", **kwargs):
    last_img_id = -1
    last_img_id_start = -1
    num_anns = len(anns)
    ann_ind = 0
    all_ind_keep = list()
    all_scores = list()
    while True:
        if ann_ind >= num_anns or anns[ann_ind]["image_id"] != last_img_id:
            if last_img_id_start >= 0:
                # DO
                num_dets = ann_ind - last_img_id_start
                dets = np.zeros((num_dets, num_parts, 3), dtype=np.float32)
                scores = np.zeros((num_dets), dtype=np.float32)
                for i in range(num_dets):
                    kps = np.array(anns[last_img_id_start+i]["keypoints"], dtype=np.float32).reshape((-1, 3))
                    dets[i] = kps
                    scores[i] = anns[last_img_id_start+i]["score"]

                if method == "nms":
                    ind_keep = np.array(nms_oks(dets, rois, scores, kwargs["nms_thres"]))
                elif method == "softnms":
                    ind_keep = np.array(softnms_oks(dets, rois, scores, sigma=kwargs["softnms_sigma"], thresh=kwargs["softnms_thres"]))
                    scores = scores[ind_keep]
                    all_scores.append(scores)

                ind_keep += last_img_id_start
                all_ind_keep.append(ind_keep)

            last_img_id_start = ann_ind

        if ann_ind >= num_anns:
            break
        last_img_id = anns[ann_ind]["image_id"]
        ann_ind += 1

    all_ind_keep = np.concatenate(all_ind_keep, axis=0)

    if method == "softnms":
        all_scores = np.concatenate(all_scores, axis=0)
        assert len(all_ind_keep) == len(all_scores)
    else:
        all_scores = None

    if return_ind:
        return all_ind_keep, all_scores

    anns = copy.deepcopy(np.array(anns)[all_ind_keep].tolist())
    if all_scores is not None:
        for ind in range(len(anns)):
            anns[ind]["score"] = all_scores[ind]
    
    return anns
