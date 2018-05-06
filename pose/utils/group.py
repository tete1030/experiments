# Functions for grouping tags
import numpy as np
from munkres import Munkres
import torch

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

class Params(object):
    def __init__(self):
        self.num_parts = 17
        self.detection_threshold = 0.2
        self.tag_threshold = 1.
        self.partOrder = [i-1 for i in [1,2,3,4,5,6,7,12,13,8,9,10,11,14,15,16,17]]
        self.max_num_people = 30
        self.use_detection_val = 0
        self.ignore_too_much = False

# Return: 
#   - pad == False:  num_people x num_parts(17) x (3+num_scale_near_one*2)
#   - pad == True :  max_num_people x num_parts(17) x (3+num_scale_near_one*2)
def match_by_tag(inp, params, pad=False):
    # tag_k: num_parts(17) x max_num_people(30) x (num_scale_near_one*2)
    # loc_k: num_parts(17) x max_num_people(30) x coord(2)
    # val_k: num_parts(17) x max_num_people(30)
    tag_k, loc_k, val_k = inp
    assert type(params) is Params

    # default_: num_parts(17) x (3+num_scale_near_one*2)
    default_ = np.zeros((params.num_parts, 3 + tag_k.shape[2]))

    dic = {}
    dic2 = {}
    for i in range(params.num_parts):
        ptIdx = params.partOrder[i]

        # tags: max_num_people(30) x (num_scale_near_one*2)
        tags = tag_k[ptIdx]
        # max_num_people(30) x coord(2)  |  max_num_people(30) x 1   |  max_num_people(30) x (num_scale_near_one*2)
        # loc | detval | tags
        # joints: max_num_people(30) x (3+num_scale_near_one*2)
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx, :, None], tags), 1)
        # joints[:, 2] is the detection value of current joint
        mask = joints[:, 2] > params.detection_threshold
        # tags: jointi_thres_num_people x (num_scale_near_one*2)
        tags = tags[mask]
        # joints: jointi_thres_num_people x (3+num_scale_near_one*2)
        joints = joints[mask]
        if i == 0 or len(dic) == 0:
            # for each person
            for tag, joint in zip(tags, joints):
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint
                dic2[tag[0]] = [tag]
        else:
            # acutalTags_key: num_first_tag x [1{tags[i][0]}]
            # actualTags: num_first_tag x [num_scale_near_one*2]
            actualTags = list(dic.keys())[:params.max_num_people]
            actualTags_key = actualTags
            # compute mean over all added joints
            actualTags = [np.mean(dic2[i], axis = 0) for i in actualTags]

            if params.ignore_too_much and len(actualTags) == params.max_num_people:
                continue

            # diff: jointi_thres_num_people x num_first_tag
            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :])**2).mean(axis = 2) ** 0.5
            if diff.shape[0]==0:
                continue

            # diff2: jointi_thres_num_people x num_first_tag
            diff2 = np.copy(diff)

            if params.use_detection_val :
                diff = np.round(diff) * 100 - joints[:, 2:3]

            # diff: jointi_thres_num_people x jointi_thres_num_people
            if diff.shape[0]>diff.shape[1]:
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10), axis = 1)

            pairs = py_max_match(-diff) ##get minimal matching
            for row, col in pairs:
                if row<diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < params.tag_threshold:
                    dic[actualTags_key[col]][ptIdx] = joints[row]
                    dic2[actualTags_key[col]].append(tags[row])
                else:
                    key = tags[row][0]
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row]
                    dic2[key] = [tags[row]]

    # ans: num_people x num_parts(17) x (3+num_scale_near_one*2)
    ans = np.array([dic[i] for i in dic])
    if pad:
        num = len(ans)
        if num < params.max_num_people:
            padding = np.zeros((params.max_num_people-num, params.num_parts, default_.shape[1]))
            if num>0: ans = np.concatenate((ans, padding), axis = 0)
            else: ans = padding
        return np.array(ans[:params.max_num_people]).astype(np.float32)
    else:
        return np.array(ans).astype(np.float32)

class HeatmapParser():
    def __init__(self, detection_val=0.03, tag_val=1., max_num_people=30):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)
        param = Params()
        param.detection_threshold = detection_val
        param.tag_threshold = tag_val
        param.ignore_too_much = True
        param.max_num_people = max_num_people
        param.use_detection_val = True
        self.param = param

    # Using max-pooling to find local maximum and set other points to 0
    # Input: det    Tensor     input array
    def nms(self, det):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    # Input: from calc result
    # Return: batch x [num_people x num_parts(17) x (3+num_scale_near_one*2)]
    def match(self, tag_k, loc_k, val_k):
        match = lambda x:match_by_tag(x, self.param)
        return list(map(match, zip(tag_k, loc_k, val_k)))

    # Find $max_num_people local minima
    # det: batch(1) x num_parts(17) x h x w
    # tag: batch(1) x num_parts(17) x h x w x (num_scale_near_one*2)
    def calc(self, det, tag):
        det = torch.autograd.Variable(torch.Tensor(det), volatile=True)
        tag = torch.autograd.Variable(torch.Tensor(tag), volatile=True)

        # find local maximum and surrounding, zero others
        det = self.nms(det)
        h = det.size()[2]
        w = det.size()[3]

        # det: batch(1) x num_parts(17) x (h*w)
        # tag: batch(1) x num_parts(17) x (h*w) x (num_scale_near_one*2)
        det = det.view(det.size()[0], det.size()[1], -1)
        tag = tag.view(tag.size()[0], tag.size()[1], det.size()[2], -1)
        # ind: batch(1) x num_parts(17) x max_num_people(30)
        # val_k: batch(1) x num_parts(17) x max_num_people(30)
        val_k, ind = det.topk(self.param.max_num_people, dim=2)
        # tag_k: batch(1) x num_parts(17) x max_num_people(30) x (num_scale_near_one*2)
        tag_k = torch.stack([torch.gather(tag[:,:,:,i], 2, ind) for i in range(tag.size()[3])], dim=3)

        x = ind % w
        y = (ind / w).long()
        # ind_k: batch(1) x num_parts(17) x max_num_people(30) x coord(2)
        ind_k = torch.stack((x, y), dim=3)
        ans = {'tag_k': tag_k, 'loc_k': ind_k, 'val_k': val_k}
        return {key:ans[key].cpu().data.numpy() for key in ans}

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2]>0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        #print(batch_id, joint_id, det[batch_id].shape)
                        tmp = det[batch_id][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
                            y+=0.25
                        else:
                            y-=0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
                            x+=0.25
                        else:
                            x-=0.25
                        ans[batch_id][people_id, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    # det: batch(1) x num_parts(17) x h x w
    # tag: batch(1) x num_parts(17) x h x w x (num_scale_near_one*2)
    # Return: batch x [num_people x num_parts(17) x (3+num_scale_near_one*2)]
    def parse(self, det, tag, adjust=True):
        ans = self.match(**self.calc(det, tag))
        if adjust:
            ans = self.adjust(ans, det)
        return ans

class FieldmapParser(object):
    def __init__(self, pair, pair_indexof, detection_thres=0.1, group_thres=0.86, max_num_people=30):
        self.pair = pair
        self.pair_indexof = pair_indexof
        self.detection_thres = detection_thres
        self.group_thres = group_thres
        self.max_num_people = max_num_people
        self.pool = torch.nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        # suppose det is a tensor
        det = torch.autograd.Variable(det, volatile=True)
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det.data

    def parse(self, det, field):
        def move_joint(ijoint_idet_2_iperson, iperson_ijoint_2_idet, iperson_A, iperson_B, mask_joint_B):
            ijoint_B_sel = torch.nonzero(mask_joint_B)[:, 0]
            idet_B_sel_ori = iperson_ijoint_2_idet[iperson_B, ijoint_B_sel]
            for ijoint_B_iter, idet_B_iter in zip(ijoint_B_sel, idet_B_sel_ori):
                ijoint_idet_2_iperson[ijoint_B_iter][idet_B_iter] = iperson_A
            iperson_ijoint_2_idet[iperson_A, ijoint_B_sel] = idet_B_sel_ori
            iperson_ijoint_2_idet[iperson_B, ijoint_B_sel] = -1

        assert isinstance(det, torch.FloatTensor) and isinstance(field, torch.FloatTensor)
        num_batch = det.size(0)
        num_joint = det.size(1)
        height = det.size(2)
        width = det.size(3)
        num_field = len(self.pair) * 2
        det = self.nms(det)
        topkval, topkind = det.view(det.size()[:2] + (-1,)).topk(self.max_num_people, dim=-1)
        topkloc = torch.stack(((topkind % width), (topkind // width)), stack=-1)

        for isample in range(num_batch):
            val_samp = []
            loc_samp = []
            force_samp = []
            idetcat_2_ijoint = []
            idetcat_2_idet = []
            ijoint_idet_2_iperson = []
            max_num_person_samp = 0
            for ijoint in range(num_joint):
                topkval_joint = topkval[isample, ijoint]
                topkloc_joint = topkloc[isample, ijoint]
                select_joint = (topkval_joint > self.detection_thres).nonzero()

                max_num_person_samp = max_num_person_samp if max_num_person_samp > len(select_joint) else len(select_joint)
                if len(select_joint) > 0:
                    val_joint = topkval_joint[select_joint[:, 0]]
                    loc_joint = topkloc_joint[select_joint[:, 0]]
                    force_joint = torch.FloatTensor(select_joint.size(0), num_field, 2)
                    for ipair, forward in self.pair_indexof[ijoint]:
                        ifield = ipair * 2 + (1 - forward)
                        # TODO: use average instead of single point
                        force_joint_x = field[isample, ifield][topkloc_joint[:, 1], topkloc_joint[:, 0]]
                        force_joint_y = field[isample, num_field+ifield][topkloc_joint[:, 1], topkloc_joint[:, 0]]
                        force_joint[:, ifield] = torch.stack([force_joint_x, force_joint_y], dim=-1)
                else:
                    val_joint = torch.FloatTensor(0)
                    loc_joint = torch.LongTensor(0)
                    force_joint = torch.FloatTensor(0)

                val_samp.append(val_joint)
                loc_samp.append(loc_joint)
                force_samp.append(force_joint)
                if len(joint_select) > 0:
                    ijoint_idet_2_iperson.append(torch.LongTensor(loc_joint.size()[:-1] + (1,)).fill_(-1))
                else:
                    ijoint_idet_2_iperson.append(torch.LongTensor(0))
                idetcat_2_ijoint.append(torch.LongTensor(len(val_joint)).fill_(ijoint))
                idetcat_2_idet.append(torch.arange(end=len(val_joint)).long())

            idetcat_2_ijoint = torch.cat(idetcat_2_ijoint, dim=0)
            idetcat_2_idet = torch.cat(idetcat_2_idet, dim=0)
            # val_samp = torch.cat(val_samp, dim=0)
            # loc_samp = torch.cat(loc_samp, dim=0)
            # loc_samp = torch.cat([loc_samp, torch.LongTensor(len(loc_samp)).fill_(-1)], dim=-1)
            # ijoint_idet_2_iperson = torch.cat(ijoint_idet_2_iperson, dim=0)
            force_samp = torch.cat(force_samp, dim=0).view(-1, 2)
            force_samp_norm = force_samp.norm(dim=-1)
            force_samp_norm_sorted, force_samp_norm_sorted_ind = force_samp_norm.sort(dim=0, descending=True)

            # TODO: sorting consider detection score

            iperson_ijoint_2_idet = torch.LongTensor(force_samp.size(0), num_joint).fill_(-1)
            counter_person = 0
            for iforce in force_samp_norm_sorted_ind:
                idetcat = iforce // num_field
                ijoint = idetcat_2_ijoint[idetcat]
                idet = idetcat_2_idet[idetcat]
                ifield = iforce % num_field
                force_field = force_samp[iforce]
                force_field_norm = force_samp_norm[iforce]
                ipair = ifield // 2
                idestend = 1 - (ifield - ipair * 2)
                iperson = ijoint_idet_2_iperson[ijoint][idet]
    
                ijoint_sec = self.pair[ipair][idestend]
                force_det = (loc_samp[ijoint_sec] - loc_samp[ijoint][idet]).float()
                similarity = (force_det[:, 0] * force_field[0] + force_det[:, 1] * force_field[1]) / (force_det.norm(dim=1) * force_field_norm)
                sim_sorted, sim_sorted_ind = similarity.sort(descending=True)
                
                # TODO: consider the distance between top cos similarities
                # TODO: consider the detection score of B detection
                # TODO: consider current person limb length: mean,variance
                
                if iperson != -1 and iperson_ijoint_2_idet[iperson, ijoint_sec] == -1:
                    for sim, idet_sec in zip(sim_sorted, sim_sorted_ind):
                        if sim < self.group_thres:
                            break
                        if ijoint_idet_2_iperson[ijoint_sec][idet_sec] != -1:
                            # TODO: consider OKS of intersection; record for combination
                            # guard: this detection should not share same iperson with A (because current person's corresponding detetion == -1)
                            iperson_sec = ijoint_idet_2_iperson[ijoint_sec][idet_sec]
                            assert iperson_sec != iperson
                            mask_joint_sec = (iperson_ijoint_2_idet[iperson_sec] > -1)
                            # if no intersection, happens when connecting part
                            if not ((iperson_ijoint_2_idet[iperson] > -1) & mask_joint_sec).any():
                                move_joint(ijoint_idet_2_iperson, iperson_ijoint_2_idet, iperson, iperson_sec, mask_joint_sec)
                                break
                            else:
                                continue
                        # A->B
                        ijoint_idet_2_iperson[ijoint_sec][idet_sec] = iperson
                        iperson_ijoint_2_idet[iperson, ijoint_sec] = idet_sec
                        break
                    # TODO: remainder process

                elif iperson == -1:
                    for sim, idet_sec in zip(sim_sorted, sim_sorted_ind):
                        if sim < self.group_thres:
                            break
                        if ijoint_idet_2_iperson[ijoint_sec][idet_sec] != -1:
                            # B->A
                            iperson_sec = ijoint_idet_2_iperson[ijoint_sec][idet_sec]
                            # if the B's correspoinding ijoint is already taken
                            if iperson_ijoint_2_idet[iperson_sec, ijoint] != -1:
                                # TODO: consider if the existed point is closer enough, and intersection between two person, and OKS of intersection; record for combination
                                # guard
                                assert iperson_ijoint_2_idet[iperson_sec, ijoint] != idet
                                continue

                            iperson = ijoint_idet_2_iperson[ijoint_sec][idet_sec]
                        else:
                            # create new person
                            iperson = counter_person
                            counter_person += 1
                            ijoint_idet_2_iperson[ijoint_sec][idet_sec] = iperson
                            # guard
                            assert iperson_ijoint_2_idet[iperson, ijoint_sec] == -1
                            iperson_ijoint_2_idet[iperson, ijoint_sec] = idet_sec

                        ijoint_idet_2_iperson[ijoint][idet] = iperson
                        # guard
                        assert iperson_ijoint_2_idet[iperson, ijoint] == -1
                        iperson_ijoint_2_idet[iperson, ijoint] = idet
                        break
                    # TODO: remainder process
                else:
                    # guard for cycle
                    assert iperson == ijoint_idet_2_iperson[ijoint_sec][iperson_ijoint_2_idet[iperson, ijoint_sec]]
            
                if iperson == -1:
                    iperson = counter_person
                    counter_person += 1
                    ijoint_idet_2_iperson[ijoint][idet] = iperson
                    iperson_ijoint_2_idet[iperson, ijoint] = idet

            iperson_sel_nonempty = ((iperson_ijoint_2_idet > -1).int().sum(dim=1) > 0).nonzero()[:, 0]
            pred = iperson_ijoint_2_idet[iperson_sel_nonempty]

            assert iperson_sel_nonempty.size(0) <= max_num_person_samp

            # TODO: Combine complementary areas (1. consider distance; 2. score by directions)
            # TODO: Combine not assigned point

            return pred
