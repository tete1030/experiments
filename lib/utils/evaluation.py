import math
import queue
import munkres
import torch
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn.linear_model import LinearRegression

from .transforms import transform, transform_preds

def calc_dists(preds, target, normalize):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n, c, 0] >= 0 and target[n, c, 1] >= 0:
                dists[c, n] = torch.dist(preds[n,c,:2], target[n,c,:2])/normalize[n]
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum().float() / dists.ne(-1).sum().float()
    else:
        return -1

def accuracy(preds, gts, head_boxes, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    norm    = 0.6*torch.from_numpy(np.linalg.norm(head_boxes.numpy()[:, 1] - head_boxes.numpy()[:, 0], axis=-1))
    dists   = calc_dists(preds, gts, norm)

    acc = torch.zeros(preds.size(1)+1, dtype=torch.float)
    avg_acc = 0
    cnt = 0

    for i in range(preds.size(1)):
        acc[i+1] = dist_acc(dists[i])
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

class OffsetCycleAverageMeter(object):
    """Computes and stores the cycled average of offset"""
    def __init__(self, size, first):
        self.size = size
        self.reset()
        self.update(first)

    def reset(self):
        self._pool = list()
        self._pointer = 0
        self.count = 0
        # self.val = None
        self.avg = None
        self.lastdiff = None
        self.avg_dir = None
        self.lastdiff_dir = None

    def update(self, val, n=1):
        for i in range(n):
            if self.count >= self.size:
                self._pool[self._pointer] = val
                self._pointer = (self._pointer + 1) % self.size
            else:
                self._pool.append(val)
                self._pointer = (self._pointer + 1) % self.size
                self.count += 1

        if self.count > 1:
            lastdiff = (val - self._pool[(self._pointer + self.count - 2) % self.count])
            avgdiff = ((val - self._pool[self._pointer % self.count]) / (self.count - 1))
            self.lastdiff_dir = lastdiff.mean().item()
            self.avg_dir = avgdiff.mean().item()
            self.lastdiff = lastdiff.abs().mean().item()
            self.avg = avgdiff.abs().mean().item()
        else:
            self.lastdiff_dir = None
            self.avg_dir = None
            self.lastdiff = None
            self.avg = None

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

def parse_map(det_map, thres=0.1):
    det_map = det_map.detach()
    if det_map.is_cuda:
        det_map = det_map.cpu()
    det_map = det_map.numpy()
    num_batch = det_map.shape[0]
    num_part = det_map.shape[1]
    height = det_map.shape[2]
    width = det_map.shape[3]

    BORDER = 10

    # det_map_border = np.zeros((num_batch, num_part, height + 2*BORDER, width + 2*BORDER))
    # det_map_border[:, :, BORDER:-BORDER, BORDER:-BORDER] = det_map

    det_map_border = det_map

    pred = np.zeros((num_batch, num_part, 3), dtype=np.float32)
    score = np.zeros((num_batch, num_part), dtype=np.float32)
    for sample_i in range(num_batch):
        for part_i in range(num_part):
            # det_map_border[sample_i, part_i] = cv2.GaussianBlur(det_map_border[sample_i, part_i], (21, 21), 0)
            loc = det_map_border[sample_i, part_i].argmax()
            y, x = np.unravel_index(loc, det_map_border.shape[-2:])
            score_sp = det_map_border[sample_i, part_i, y, x]

            # det_map_border[sample_i, part_i, y, x] = 0
            # loc2 = det_map_border[sample_i, part_i].argmax()
            # y2, x2 = np.unravel_index(loc2, det_map_border.shape[-2:])

            # y2 -= y
            # x2 -= x
            # ln = (x2 ** 2 + y2 ** 2) ** 0.5
            # delta = 0.25
            # if ln > 1e-3:
            #     x += delta * x2 / ln
            #     y += delta * y2 / ln

            if det_map_border[sample_i, part_i, y, max(0, x-1)] < det_map_border[sample_i, part_i, y, min(width-1, x+1)]:
                off_x = 0.25
            else:
                off_x = -0.25
            if det_map_border[sample_i, part_i, max(0, y-1), x] < det_map_border[sample_i, part_i, min(height-1, y+1), x]:
                off_y = 0.25
            else:
                off_y = -0.25

            # y -= BORDER
            # x -= BORDER
            x = max(0, min(x, det_map.shape[3] - 1))
            y = max(0, min(y, det_map.shape[2] - 1))

            pred[sample_i, part_i, 0] = x + 0.5 + off_x
            pred[sample_i, part_i, 1] = y + 0.5 + off_y
            pred[sample_i, part_i, 2] = 1
            score[sample_i, part_i] = score_sp

    return pred, score

def generate_ans(image_ids, preds, scores, det_roi_scores=None, det_roi_use="no"):
    ans = []
    for sample_i in range(len(preds)):
        image_id = image_ids[sample_i]

        val = preds[sample_i]
        if det_roi_use == "no":
            score = scores[sample_i].mean()
        elif det_roi_use == "avg":
            score = (scores[sample_i].sum() + det_roi_scores[sample_i]) / (scores.shape[1] + 1)
        elif det_roi_use == "mul":
            score = scores[sample_i].mean() * det_roi_scores[sample_i]
        else:
            raise ValueError()
        tmp = {'image_id':int(image_id), "category_id": 1, "keypoints": [], "score":float(score)}
        # # p: average detected locations
        # p = val[val[:, 2] > 0][:, :2].mean(axis = 0)
        # for j in val:
        #     if j[2]>0.:
        #         tmp["keypoints"] += [float(j[0]), float(j[1]), 1]
        #     else:
        #         # TRICK: for not detected points, place them at the average point
        #         tmp["keypoints"] += [float(p[0]), float(p[1]), 0]
        tmp["keypoints"] = val.ravel().tolist()
        ans.append(tmp)
    return ans

def generate_mpii_ans(image_ids, person_ids, preds):
    assert len(image_ids) == len(person_ids) and len(person_ids) == len(preds)
    return {"image_ids": image_ids, "person_ids": person_ids, "preds": preds}
