import numpy as np
import torch
import math
from torch import nn
from torch import autograd
from .simphg import Conv, Pool, SimpHourglass
import torchvision.models as tvmodels

try:
    profile
except NameError:
    profile = lambda func: func

__all__ = ["PoseManager", "PoseMapParser", "PoseNet", "PoseMapLoss", "PoseDisLoss", "MergeResNet"]

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseManager(object):
    def __init__(self, batch_size, num_parts, out_size, max_num_people, cuda=True, sigma=1, filter_inside=True, gen_embedding=True):
        """Generate Pose Map and s Map

        Arguments:
            batch_size {int} -- Batch size
            num_parts {int} -- Numble of parts
            out_size {tuple} -- height x width
            max_num_people {int} -- Max number of people

        Keyword Arguments:
            cuda {bool} -- Using CUDA (default: {True})
            sigma {int} -- Sigma of point (default: {1})
        """

        # size: batch_size x (num_parts*2) x h x w
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.out_size = out_size
        self.max_num_people = max_num_people
        self.cuda = cuda
        self.sigma = sigma
        self.filter_inside = filter_inside
        self.gen_embedding = gen_embedding

        if self.gen_embedding:
            self.map = torch.zeros(
                batch_size, num_parts*2, *out_size)
        else:
            self.map = torch.zeros(
                batch_size, num_parts, *out_size)
        temp_size = 6*sigma + 3
        x = np.arange(0, temp_size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        keypoints_temp = np.exp(- ((x - x0) ** 2 +
                                   (y - y0) ** 2) / (2 * sigma ** 2))
        embedding_temp = (np.sqrt(((x - x0) ** 2 + (y - y0) ** 2))
                          <= (3*sigma + 1)).astype(np.float32)
        self.keypoints_temp = torch.from_numpy(keypoints_temp * embedding_temp)
        self.embedding_temp = torch.from_numpy(embedding_temp)
        self.temp_length = temp_size * temp_size
        assert self.temp_length % 2 == 1
        self.temp_middle = (self.temp_length + 1) / 2 - 1

        if self.cuda:
            self.map = self.map.cuda(async=True)
            self.keypoints_temp = self.keypoints_temp.cuda(async=True)
            self.embedding_temp = self.embedding_temp.cuda(async=True)

        self.cur_batch_size = None
        self.all_split = None
        self.all_keypoints_var = None
        self.all_keypoints_info = None

        self.draw_rows_map = None
        self.draw_cols_map = None
        self.draw_rows_temp = None
        self.draw_cols_temp = None
        self.draw_insider = None

        self.batch_ids = None
        self.person_cat_ids = None
        self.person_ids = None
        self.person_mask = None
        self.part_ids = None

    def init_with_locate(self, locate):
        self.init(locate, True)

    def init_with_keypoints(self, keypoints):
        self.init(keypoints, False)

    @profile
    def init(self, keypoints, is_locate):
        """Init flat keypoints

        Arguments:
            keypoints {list} -- #batch x [#batch_i_person x #part x 3] or #batch x [#batch_i_person x 2]
        """
        # num_batch could be smaller than self.batch_size when tailer batch
        num_batch = len(keypoints)
        assert num_batch <= self.batch_size

        num_person_list = torch.LongTensor([len(k) for k in keypoints])
        split = torch.cumsum(num_person_list, dim=0).tolist()

        # keypoints_cat: #all_person x #part x 3 or #all_person x 2
        keypoints_cat = torch.cat(keypoints, dim=0).float()
        assert len(keypoints_cat) > 0

        if is_locate:
            keypoints_cat = keypoints_cat[:, :2].view(-1, 1, 2).repeat(1, self.num_parts, 1)
            keypoints_info = torch.LongTensor(keypoints_cat.size(0), self.num_parts).fill_(1)
        else:
            keypoints_info = keypoints_cat[:, :, 2].long()
            keypoints_cat = keypoints_cat[:, :, :2]

        if self.cuda:
            keypoints_cat = keypoints_cat.cuda(async=True)
            keypoints_info = keypoints_info.cuda(async=True)

        batch_all_ids = torch.LongTensor(split[-1])
        person_all_ids = torch.LongTensor(split[-1])
        person_ids_split2cat = torch.LongTensor(num_batch, num_person_list.max())

        if self.cuda:
            batch_all_ids = batch_all_ids.cuda(async=True)
            person_all_ids = person_all_ids.cuda(async=True)
            person_ids_split2cat = person_ids_split2cat.cuda(async=True)

        last_end_pos = 0
        for ib in range(num_batch):
            end_pos = split[ib]
            if end_pos > last_end_pos:
                batch_all_ids[last_end_pos:end_pos] = ib
                person_all_ids[last_end_pos:end_pos] = torch.arange(end=num_person_list[ib]).long()
                person_ids_split2cat[ib, :num_person_list[ib]] = torch.arange(start=last_end_pos, end=last_end_pos+num_person_list[ib]).long()
            last_end_pos = end_pos

        # Filter out part not labeled
        # cat_ids, part_ids: #valid_part
        cat_ids, part_ids = torch.nonzero(keypoints_info > 0).t()
        batch_ids = batch_all_ids[cat_ids]
        person_ids = person_all_ids[cat_ids]
        max_person_id = person_ids.max()
        person_mask = [(person_ids == i)
                       for i in range(max_person_id)]
        part_ids = part_ids
        person_cat_ids = person_ids_split2cat[batch_ids, person_ids]

        # Filter and flatten result
        # keypoints_flat: #valid_part x 2
        keypoints_flat = keypoints_cat[cat_ids, part_ids]
        keypoints_flat_x = keypoints_flat[:, 0]
        keypoints_flat_y = keypoints_flat[:, 1]

        cols, rows = np.meshgrid(
                np.arange(-3*self.sigma-1, 3*self.sigma+2, dtype=np.float32),
                np.arange(-3*self.sigma-1, 3*self.sigma+2, dtype=np.float32), indexing="xy")

        cols = torch.from_numpy(cols).float()
        rows = torch.from_numpy(rows).float()
        if self.cuda:
            cols = cols.cuda(async=True)
            rows = rows.cuda(async=True)
        cols = cols.view(-1, 1).expand(-1, batch_ids.size(0))
        rows = rows.view(-1, 1).expand(-1, batch_ids.size(0))

        draw_cols_map = cols + keypoints_flat_x
        draw_rows_map = rows + keypoints_flat_y
        draw_cols_temp = cols.long() + 3*self.sigma + 1
        draw_rows_temp = rows.long() + 3*self.sigma + 1
        draw_insider = ((draw_cols_map >= 0) & (draw_cols_map < self.out_size[1]) & \
                        (draw_rows_map >= 0) & (draw_rows_map < self.out_size[0]))

        self.cur_batch_size = num_batch

        self.all_split = split
        # self.all_keypoints_var: #all_person x #part x 2
        self.all_keypoints_var = torch.autograd.Variable(keypoints_cat)
        # self.all_keypoints_info: #all_person x #part
        self.all_keypoints_info = keypoints_info

        self.draw_rows_map = draw_rows_map
        self.draw_cols_map = draw_cols_map
        self.draw_rows_temp = draw_rows_temp
        self.draw_cols_temp = draw_cols_temp
        self.draw_insider = draw_insider

        self.batch_ids = batch_ids
        self.person_ids = person_ids
        self.person_mask = person_mask
        self.person_cat_ids = person_cat_ids
        self.part_ids = part_ids

        if self.filter_inside:
            self.filter_valid_point()

    @profile
    def filter_valid_point(self):
        insider = self.draw_insider[self.temp_middle]
        outsider = ~insider
        insider = torch.nonzero(insider)
        outsider = torch.nonzero(outsider)

        if len(outsider) > 0:
            outsider = outsider[:, 0]
            out_person_cat_ids = self.person_cat_ids[outsider]
            out_part_ids = self.part_ids[outsider]

            # Mark these keypoints as NOT LABELED
            self.all_keypoints_info[out_person_cat_ids, out_part_ids] = 0

        if len(insider) > 0:
            insider = insider[:, 0]
            self.batch_ids = self.batch_ids[insider]
            self.person_ids = self.person_ids[insider]
            self.person_cat_ids = self.person_cat_ids[insider]
            self.part_ids = self.part_ids[insider]
            self.person_mask = [pm[insider] for pm in self.person_mask]

            self.draw_rows_map = self.draw_rows_map[:, insider]
            self.draw_cols_map = self.draw_cols_map[:, insider]
            self.draw_rows_temp = self.draw_rows_temp[:, insider]
            self.draw_cols_temp = self.draw_cols_temp[:, insider]
            self.draw_insider = self.draw_insider[:, insider]
        else:
            self.batch_ids = torch.LongTensor(0).cuda(async=True)
            self.person_ids = torch.LongTensor(0).cuda(async=True)
            self.person_cat_ids = torch.LongTensor(0).cuda(async=True)
            self.part_ids = torch.LongTensor(0).cuda(async=True)
            self.person_mask = [torch.ByteTensor(0).cuda(async=True) for pm in self.person_mask]

            self.draw_rows_map = torch.FloatTensor(0).cuda(async=True)
            self.draw_cols_map = torch.FloatTensor(0).cuda(async=True)
            self.draw_rows_temp = torch.LongTensor(0).cuda(async=True)
            self.draw_cols_temp = torch.LongTensor(0).cuda(async=True)
            self.draw_insider = torch.ByteTensor(0).cuda(async=True)

    @profile
    def move_keypoints(self, movement_x, movement_y):
        """Move keypoints
        """
        if len(self.batch_ids) == 0:
            return

        assert isinstance(movement_x, autograd.Variable) and isinstance(movement_y, autograd.Variable)

        self.draw_rows_map += movement_y.data
        self.draw_cols_map += movement_x.data
        self.draw_insider = ((self.draw_cols_map >= 0) & (self.draw_cols_map < self.out_size[1]) & \
                             (self.draw_rows_map >= 0) & (self.draw_rows_map < self.out_size[0]))

        selector_x = torch.LongTensor([0]).cuda(async=True).expand(len(self.batch_ids))
        selector_y = torch.LongTensor([1]).cuda(async=True).expand(len(self.batch_ids))

        self.all_keypoints_var[self.person_cat_ids,
                               self.part_ids,
                               selector_x] += movement_x
        self.all_keypoints_var[self.person_cat_ids,
                               self.part_ids,
                               selector_y] += movement_y
        if self.filter_inside:
            self.filter_valid_point()

    def extract_movement(self, move_field=None, factor=None, move_vector=None, embedding=None):
        assert bool(move_field is not None) != bool(move_vector is not None)
        assert bool(move_field is not None) == bool(factor is not None)
        assert bool(move_vector is not None) == bool(embedding is not None)

        if move_field is not None:
            rows = self.draw_rows_map[self.temp_middle].long() / int(factor)
            cols = self.draw_cols_map[self.temp_middle].long() / int(factor)

            movement_x = move_field[self.batch_ids,
                                    self.part_ids,
                                    rows,
                                    cols]
            movement_y = move_field[self.batch_ids,
                                    self.part_ids + self.num_parts,
                                    rows,
                                    cols]
        else:
            vec_ids = (embedding[self.person_ids, self.batch_ids].long() - 1) * self.num_parts + self.part_ids

            movement_x = move_vector[self.batch_ids, vec_ids]
            movement_y = move_vector[self.batch_ids, vec_ids + self.max_num_people * self.num_parts]

        return movement_x, movement_y

    def generate_embedding(self):
        # embedding: #batch x max_num_people
        embedding = []
        for i in range(self.cur_batch_size):
            emb = np.arange(1, self.max_num_people+1)
            np.random.shuffle(emb)
            embedding.append(emb)
        # embedding: max_num_people x #batch
        embedding = torch.from_numpy(np.array(embedding, dtype=np.float32).T).contiguous()
        if self.cuda:
            embedding = embedding.cuda(async=True)

        return embedding

    @profile
    def generate(self, embedding):
        self.map.zero_()

        if self.cur_batch_size == 0:
            return torch.FloatTensor(0).cuda(async=True)

        if len(self.batch_ids) == 0:
            return self.map[:self.cur_batch_size].clone()

        batch_ids = self.batch_ids.expand(self.temp_length, -1)
        person_mask = [pm.expand(self.temp_length, -1) for pm in self.person_mask]
        part_ids = self.part_ids.expand(self.temp_length, -1)

        map_cols = self.draw_cols_map
        map_rows = self.draw_rows_map
        temp_cols = self.draw_cols_temp
        temp_rows = self.draw_rows_temp
        insider = self.draw_insider

        # select pixels inside
        batch_ids = batch_ids[insider]
        part_ids = part_ids[insider]
        map_rows = map_rows[insider].long()
        map_cols = map_cols[insider].long()
        temp_rows = temp_rows[insider]
        temp_cols = temp_cols[insider]
        person_mask = [mask_pi[insider] for mask_pi in person_mask]

        for i, mask_pi in enumerate(person_mask):
            # mask_pi: #all_parts x temp_size

            # Select point overlapped with other points but nearer to current part
            # mask_modify: #all_parts x temp_size

            mask_modify = self.map[batch_ids, part_ids, map_rows, map_cols]\
                .lt(self.keypoints_temp[temp_rows, temp_cols])

            mask = mask_modify & mask_pi

            batch_ids_masked = batch_ids[mask]
            part_ids_masked = part_ids[mask]
            map_rows_masked = map_rows[mask]
            map_cols_masked = map_cols[mask]
            temp_rows_masked = temp_rows[mask]
            temp_cols_masked = temp_cols[mask]

            if len(batch_ids_masked) > 0:
                self.map[batch_ids_masked, part_ids_masked, map_rows_masked, map_cols_masked] = \
                    self.keypoints_temp[temp_rows_masked, temp_cols_masked]

                if self.gen_embedding:
                    self.map[batch_ids_masked, part_ids_masked+self.num_parts, map_rows_masked, map_cols_masked] = \
                        self.embedding_temp[temp_rows_masked, temp_cols_masked] * embedding[i][batch_ids_masked]

        return self.map[:self.cur_batch_size].clone()

    def get_split_keypoints(self):
        split = self.all_split
        keypoint_cat = self.all_keypoints_var.data.cpu()
        keypoint_cat_info = self.all_keypoints_info.cpu()
        keypoint = [torch.cat((keypoint_cat[start:end],
                               keypoint_cat_info[start:end]
                               .contiguous().float().view(end-start, self.num_parts, 1)),
                              dim=2) if end > start else torch.FloatTensor(0)
                        for start, end in zip([0] + split[:-1], split)]
        return keypoint

class PoseHGModule(nn.Module):
    def __init__(self, inp_dim, out_dim, merge, bn=False, increase=128):
        super(PoseHGModule, self).__init__()
        self.features = nn.Sequential(
            SimpHourglass(inp_dim, n=4, bn=bn, increase=increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        )
        self.outs = Conv(inp_dim, out_dim, 1, relu=False, bn=False)
        if merge:
            self.merge_features = Merge(inp_dim, inp_dim)
            self.merge_preds = Merge(out_dim, inp_dim)

    def forward(self, x):
        feature = self.features(x)
        pred = self.outs(feature)
        if self.merge_features:
            x = x + self.merge_preds(pred) + self.merge_features(feature)
        return pred, x

# Partly adopted from pose-ae-train
class PoseMapParser():
    def __init__(self, cuda, nms_kernel_size=3, threshold=0.03):
        self.pool = nn.MaxPool2d(nms_kernel_size, 1, 1)
        if cuda:
            self.pool = self.pool.cuda()
        self.threshold = threshold

    def is_local_max(self, det, mask):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        if mask is not None:
            maxm *= mask[:, None, :, :].float()
        det = det * maxm
        return det >= self.threshold

    def parse(self, det, mask, factor):
        """from detection map to keypoints

        Arguments:
            det {Tensor/Variable} -- detection map: #batch x 1 x h/factor x w/factor
            mask {Tensor/Variable} -- mask: #batch x h/factor x w/factor
            factor {int} -- factor from map position to coordination
        """

        det_lm = self.is_local_max(det, mask)
        det_ids = det_lm.nonzero()

        if len(det_ids) == 0:
            return [torch.LongTensor(0) for i in range(det.size(0))]
        if isinstance(det_ids, torch.autograd.Variable):
            det_ids = det_ids.data
        if det_ids.is_cuda:
            det_ids = det_ids.cpu()

        det_bid = [torch.nonzero(det_ids[:, 0] == i) for i in range(det.size(0))]
        det_pid = det_ids[:, 1]
        det_pos = det_ids[:, [3, 2]] * factor

        if det.size(1) == 1:
            return [det_pos[dbid[:, 0]] if len(dbid) > 0 else torch.LongTensor(0)
                    for dbid in det_bid]
        else:
            return [(det_pid[dbid[:, 0]], det_pos[dbid[:, 0]]) if len(dbid) > 0 else torch.LongTensor(0)
                    for dbid in det_bid]

class PoseMapLoss(nn.Module):
    def __init__(self):
        super(PoseMapLoss, self).__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        l = ((pred - gt)**2) * mask[:, None, :, :].expand_as(pred).float()
        l = l.mean()
        return l

class PoseDisLoss(nn.Module):
    def __init__(self):
        super(PoseDisLoss, self).__init__()

    def forward(self, kp_pred, kp_gt, masks, locate_std_gt):
        """Calculate move field loss

        Arguments:
            kp_pred {Tensor} -- #all_person x #part x 2
            kp_gt {Tensor} -- #all_person x #part x 2
            masks {Tensor} -- #all_person x #part
            locate_std_gt -- #all_person
        """
        assert kp_pred.dim() == 3
        assert kp_pred.size() == kp_gt.size()
        loss = (((kp_pred.float() - kp_gt.float()) ** 2).sum(dim=-1) * masks.float()).sum(dim=-1) / locate_std_gt
        loss = (loss.sum() / masks.float().sum()).sqrt()

        return loss

class PosePre(nn.Module):
    def __init__(self, img_dim, hg_dim, bn=False):
        super(PosePre, self).__init__()
        self.pre = nn.Sequential(
            Conv(img_dim, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, hg_dim, bn=bn)
        )

    def forward(self, imgs):
        x = self.pre(imgs)
        return x

class PoseNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hg_dim=256, merge_inp_dim=0, bn=False):
        super(PoseNet, self).__init__()
        self.pre = PosePre(img_dim=inp_dim, hg_dim=hg_dim, bn=bn)
        if merge_inp_dim > 0:
            self.merge_pre = PosePre(img_dim=merge_inp_dim, hg_dim=hg_dim, bn=bn)
            self.merge_data = None
        self.hgmod = PoseHGModule(inp_dim=hg_dim, out_dim=out_dim, merge=True, bn=bn)

    def forward(self, inp=None, merge_inp=None, merge=False, free_merge=False):
        assert inp is not None or merge_inp is not None or free_merge
        assert not (inp is None and merge)
        # Indicating data not properly freed
        assert merge_inp is not None and self.merge_data is not None

        if merge_inp is not None:
            self.merge_data = self.merge_pre(merge_inp)

        if inp is not None:
            if merge:
                assert self.merge_data is not None
                ret = self.hgmod(self.pre(inp) + self.merge_data)
            else:
                ret = self.hgmod(self.pre(inp))
        else:
            ret = None

        if free_merge:
            self.merge_data = None

        return ret

class MergeResNet(nn.Module):
    def __init__(self, inp_size, num_classes):
        super(MergeResNet, self).__init__()
        inp_resnet = tvmodels.resnet50(num_classes=num_classes)
        merge_resnet = tvmodels.resnet50(num_classes=num_classes)
        inp_resnet_conv1 = nn.Conv2d(inp_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
        inp_resnet_conv1_para_n = inp_resnet_conv1.kernel_size[0] * inp_resnet_conv1.kernel_size[1] * inp_resnet_conv1.out_channels
        inp_resnet_conv1.weight.data.normal_(0, math.sqrt(2. / inp_resnet_conv1_para_n))

        self.inp_pre = nn.Sequential(inp_resnet_conv1, inp_resnet.bn1, inp_resnet.relu, inp_resnet.maxpool)
        self.merge_pre = nn.Sequential(merge_resnet.conv1, merge_resnet.bn1, merge_resnet.relu, merge_resnet.maxpool)
        self.layers = nn.Sequential(inp_resnet.layer1, inp_resnet.layer2, inp_resnet.layer3, inp_resnet.layer4)
        self.avgpool = inp_resnet.avgpool
        # self.fc = inp_resnet.fc
        self.fc = nn.Linear(8192, num_classes)
        self.merge_data = None
    
    def res_forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, inp=None, merge_inp=None, merge=False, free_merge=False):
        assert inp is not None or merge_inp is not None or free_merge
        assert not (inp is None and merge)
        # Indicating data not properly freed
        assert not (merge_inp is not None and self.merge_data is not None)

        if merge_inp is not None:
            self.merge_data = self.merge_pre(merge_inp)

        if inp is not None:
            if merge:
                assert self.merge_data is not None
                ret = self.res_forward(self.inp_pre(inp) + self.merge_data)
            else:
                ret = self.res_forward(self.inp_pre(inp))
        else:
            ret = None

        if free_merge:
            self.merge_data = None

        return ret