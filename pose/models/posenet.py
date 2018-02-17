import numpy as np
import torch
from torch import nn
from torch import autograd
from .simphg import Conv, Pool, SimpHourglass

__all__ = ["PoseManager", "PoseMapParser", "PoseNet", "PoseMapLoss", "PoseDisLoss"]

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseManager(object):
    def __init__(self, batch_size, num_parts, out_size, max_num_people, cuda=True, sigma=1):
        """Generate Pose Map and Embedding Map

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

        self.map = torch.zeros(
            batch_size, num_parts*2, *out_size)
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

        if self.cuda:
            self.map = self.map.cuda()
            self.keypoints_temp = self.keypoints_temp.cuda()
            self.embedding_temp = self.embedding_temp.cuda()

        self.cur_batch_size = None
        self.all_split = None
        self.all_keypoints_var = None
        self.all_keypoints_info = None
        self.draw_batch_ids = None
        self.draw_person_ids = None
        self.draw_person_cat_ids = None
        self.draw_part_ids = None
        self.draw_keypoints_flat = None

    def init_with_locate(self, locate):
        self.init(locate, True)

    def init_with_keypoints(self, keypoints):
        self.init(keypoints, False)
    
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
        keypoints_cat = torch.cat(keypoints, dim=0)

        batch_all_ids = torch.LongTensor(split[-1])
        person_all_ids = torch.LongTensor(split[-1])
        person_ids_split2cat = torch.LongTensor(num_batch, num_person_list.max())

        last_end_pos = 0
        for ib in range(num_batch):
            end_pos = split[ib]
            if end_pos > last_end_pos:
                batch_all_ids[last_end_pos:end_pos] = ib
                person_all_ids[last_end_pos:end_pos] = torch.arange(end=num_person_list[ib]).long()
                person_ids_split2cat[ib, :num_person_list[ib]] = torch.arange(end=num_person_list[ib]).long() + last_end_pos
            last_end_pos = end_pos

        if not is_locate:
            # Filter out part not labeled
            # cat_ids, part_ids: #valid_part
            cat_ids, part_ids = torch.nonzero(keypoints_cat[:, :, 2] >= 1).t()
            batch_ids = batch_all_ids[cat_ids]
            person_ids = person_all_ids[cat_ids]
            person_cat_ids = person_ids_split2cat[batch_ids, person_ids]

            keypoints_info = keypoints_cat[:, :, 2]
            keypoints_cat = keypoints_cat[:, :, :2]
            # Filter and flatten result
            # keypoints_flat: #valid_part x 2
            keypoints_flat = keypoints_cat[cat_ids, part_ids].long()
        else:
            batch_ids = batch_all_ids.view(-1, 1).repeat(1, self.num_parts).view(-1)
            person_ids = person_all_ids.view(-1, 1).repeat(1, self.num_parts).view(-1)
            part_ids = torch.arange(end=self.num_parts).long().repeat(person_all_ids.size(0))
            person_cat_ids = person_ids_split2cat[batch_ids, person_ids]

            keypoints_cat = keypoints_cat.view(-1, 1, 2).repeat(1, self.num_parts, 1)
            keypoints_info = torch.ones(*keypoints_cat.size()[:2]).long()
            keypoints_flat = keypoints_cat.view(-1, 2).long()

        self.cur_batch_size = num_batch

        self.all_split = split
        # self.all_keypoints_var: #all_person x #part x 2
        self.all_keypoints_var = torch.autograd.Variable(keypoints_cat)
        if self.cuda:
            self.all_keypoints_var = self.all_keypoints_var.cuda()
        # self.all_keypoints_info: #all_person x #part
        self.all_keypoints_info = keypoints_info

        self.draw_batch_ids = batch_ids.contiguous()
        self.draw_person_ids = person_ids.contiguous()
        self.draw_person_cat_ids = person_cat_ids.contiguous()
        self.draw_part_ids = part_ids.contiguous()
        self.draw_keypoints_flat = keypoints_flat

        self.filter_valid_point()

    def filter_valid_point(self):
        kpx = self.draw_keypoints_flat[:, 0]
        kpy = self.draw_keypoints_flat[:, 1]
        insider = ((kpx >= 0) & (kpx < self.out_size[1]) & \
                   (kpy >= 0) & (kpy < self.out_size[0]))
        outsider = ~insider

        batch_ids = self.draw_batch_ids[insider].contiguous()
        person_ids = self.draw_person_ids[insider].contiguous()
        person_cat_ids = self.draw_person_cat_ids[insider].contiguous()
        part_ids = self.draw_part_ids[insider].contiguous()
        insider_pos = torch.nonzero(insider)
        if len(insider_pos) > 0:
            keypoints_flat = self.draw_keypoints_flat[insider_pos[:, 0]]
        else:
            keypoints_flat = torch.LongTensor(0)

        # out_batch_ids = self.draw_batch_ids[outsider].contiguous()
        # out_person_ids = self.draw_person_ids[outsider].contiguous()
        out_person_cat_ids = self.draw_person_cat_ids[outsider].contiguous()
        out_part_ids = self.draw_part_ids[outsider].contiguous()
        # out_keypoints_flat = self.draw_keypoints_flat[torch.nonzero(outsider)[:, 0]]

        # Mark these keypoints as NOT LABELED
        if len(out_person_cat_ids) > 0:
            # TODO: EFFICIENCY use cuda from beginning
            # TODO: COMPATIBILITY detect if cuda enabled
            self.all_keypoints_info[out_person_cat_ids, out_part_ids] = 0
        
        self.draw_batch_ids = batch_ids
        self.draw_person_ids = person_ids
        self.draw_person_cat_ids = person_cat_ids
        self.draw_part_ids = part_ids
        self.draw_keypoints_flat = keypoints_flat
        # return out_batch_ids, out_person_ids, out_part_ids, out_keypoints_flat

    def move_keypoints(self, move_field, factor=1):
        """Move keypoints along vectors in move_field
        
        Arguments:
            move_field {Tensor} -- #batch x #parts x h x w
            factor {int} -- ratio of keypoints to move_field
        """
        if len(self.draw_batch_ids) == 0:
            return

        # TODO: EFFICIENCY use cuda once, or from beginning
        batch_ids = self.draw_batch_ids.cuda()
        part_ids = self.draw_part_ids.cuda()
        rows = (self.draw_keypoints_flat[:, 1] / factor).long().cuda()
        cols = (self.draw_keypoints_flat[:, 0] / factor).long().cuda()
        # TODO: COMPATIBILITY detect if cuda enabled
        movement_x = move_field[batch_ids,
                                part_ids,
                                rows,
                                cols]
        movement_y = move_field[batch_ids,
                                part_ids + self.num_parts,
                                rows,
                                cols]

        if not move_field.is_cuda:
            self.draw_keypoints_flat[:, 0] += movement_x.data.long()
            self.draw_keypoints_flat[:, 1] += movement_y.data.long()
        else:
            self.draw_keypoints_flat[:, 0] += movement_x.cpu().data.long()
            self.draw_keypoints_flat[:, 1] += movement_y.cpu().data.long()

        # TODO: EFFICIENCY change to efficient way
        selector_x = torch.LongTensor(len(self.draw_batch_ids)).fill_(0)
        selector_y = torch.LongTensor(len(self.draw_batch_ids)).fill_(1)

        # TODO: EFFICIENCY use cuda from beginning
        # TODO: COMPATIBILITY detecting cuda option
        self.all_keypoints_var[self.draw_person_cat_ids.cuda(),
                               self.draw_part_ids.cuda(),
                               selector_x.cuda()] += movement_x.long()
        self.all_keypoints_var[self.draw_person_cat_ids.cuda(),
                               self.draw_part_ids.cuda(),
                               selector_y.cuda()] += movement_y.long()
        self.filter_valid_point()

    def generate(self):
        self.map.zero_()

        if self.cur_batch_size == 0:
            return autograd.Variable(torch.FloatTensor(0).cuda())

        if len(self.draw_batch_ids) == 0:
            return self.map[:self.cur_batch_size].clone()

        # batch_ids, person_ids, part_ids: #all_parts
        # keypoints_flat: #all_parts x 2
        batch_ids = self.draw_batch_ids
        person_ids = self.draw_person_ids
        part_ids = self.draw_part_ids
        keypoints_flat = self.draw_keypoints_flat

        temp_size = self.keypoints_temp.size(0) * self.keypoints_temp.size(1)

        max_person_id = person_ids.max()
        person_id_masks = [(person_ids == i).view(-1, 1).repeat(1, temp_size)
                           for i in range(max_person_id)]

        max_batch_id = batch_ids.max()

        batch_ids = batch_ids.view(-1, 1).repeat(1, temp_size)
        # person_ids = person_ids.view(-1, 1).repeat(1, temp_size)
        part_ids = part_ids.view(-1, 1).repeat(1, temp_size)

        # loc ~ (X, Y)
        # loc: 2 x temp_height x temp_width
        loc = torch.from_numpy(np.array(
            np.meshgrid(
                range(-3*self.sigma-1, 3*self.sigma+2),
                range(-3*self.sigma-1, 3*self.sigma+2), indexing="xy")))
        # loc: 2 x #all_parts x temp_size
        loc = loc.view(2, 1, -1).expand(2, batch_ids.size(0), -1)
        map_loc = loc + keypoints_flat.t().contiguous().view(2, -1, 1).expand(-1, -1, temp_size)
        map_cols = map_loc[0]
        map_rows = map_loc[1]
        temp_loc = loc + 3*self.sigma + 1
        temp_cols = temp_loc[0]
        temp_rows = temp_loc[1]

        # insider: #all_parts x temp_size
        insider = ((map_cols >= 0) & (map_cols < self.out_size[1]) & \
                   (map_rows >= 0) & (map_rows < self.out_size[0]))

        # select points inside
        # TODO: EFFICIENCY use cuda from beginning
        # TODO: EFFICIENCY async
        batch_ids = batch_ids[insider].cuda(async=False)
        part_ids = part_ids[insider].cuda(async=False)
        map_rows = map_rows[insider].cuda(async=False)
        map_cols = map_cols[insider].cuda(async=False)
        temp_rows = temp_rows[insider].cuda(async=False)
        temp_cols = temp_cols[insider].cuda(async=False)
        person_id_masks = [mask_pi[insider].cuda(async=False) for mask_pi in person_id_masks]

        # embeddings: #batch x max_num_people
        embeddings = []
        for i in range(max_batch_id+1):
            emb = np.arange(1, self.max_num_people+1)
            np.random.shuffle(emb)
            embeddings.append(emb)
        # embeddings: max_num_people x #batch
        embeddings = torch.from_numpy(np.array(embeddings).T).float().cuda(async=False)

        for i, mask_pi in enumerate(person_id_masks):
            # mask_pi: #all_parts x temp_size

            # Select point overlapped with other points but nearer to current part
            # mask_modify: #all_parts x temp_size
            # TODO: EFFICIENCY change to memory efficient way
            mask_modify = torch.lt(self.map[batch_ids, part_ids, map_rows, map_cols],
                                   self.keypoints_temp[temp_rows, temp_cols])
            mask = (mask_pi & mask_modify)

            # #point
            bids_masked = batch_ids[mask]
            pids_masked = part_ids[mask]
            map_row_masked = map_rows[mask]
            map_col_masked = map_cols[mask]
            temp_row_masked = temp_rows[mask]
            temp_col_masked = temp_cols[mask]

            if len(bids_masked) > 0:
                self.map[bids_masked, pids_masked, map_row_masked, map_col_masked] = \
                    self.keypoints_temp[temp_row_masked, temp_col_masked]

                self.map[bids_masked, pids_masked+self.num_parts, map_row_masked, map_col_masked] = \
                    self.embedding_temp[temp_row_masked, temp_col_masked] * embeddings[i][bids_masked]

        return self.map[:self.cur_batch_size].clone()

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

    def is_local_max(self, det):
        # suppose det is a tensor
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det >= self.threshold

    def parse(self, det, factor):
        """from detection map to keypoints
        
        Arguments:
            det {Tensor/Variable} -- detection map: #batch x 1 x h/factor x w/factor
            factor {int} -- factor from map position to coordination
        """

        det_lm = self.is_local_max(det)
        det_ids = det_lm.nonzero() * factor
        if len(det_ids) == 0:
            return [torch.LongTensor(0) for i in range(det.size(0))]
        if isinstance(det_ids, autograd.Variable):
            det_ids = det_ids.data
        if det_ids.is_cuda:
            det_ids = det_ids.cpu()
        det_pos = det_ids[:, [3, 2]]
        det_bid = [torch.nonzero(det_ids[:, 0] == i) for i in range(det.size(0))]
        return [det_pos[dbid[:, 0]] if len(dbid) > 0 else torch.LongTensor(0)
                for dbid in det_bid]

class PoseMapLoss(nn.Module):
    def __init__(self):
        super(PoseMapLoss, self).__init__()

    def forward(self, pred, gt, masks):
        assert pred.size() == gt.size()
        l = ((pred - gt)**2) * masks[:, None, :, :].expand_as(pred).float()
        l = l.mean()
        return l

class PoseDisLoss(nn.Module):
    def __init__(self):
        super(PoseDisLoss, self).__init__()

    def forward(self, kp_pred, kp_gt, masks):
        """Calculate move field loss

        Arguments:
            kp_pred {Tensor} -- #all_person x #part x 2
            kp_gt {Tensor} -- #all_person x #part x 2
        """
        assert kp_pred.dim() == 3
        assert kp_pred.size() == kp_gt.size()
        loss = ((kp_pred.float() - kp_gt.float()) ** 2).sum(dim=-1) * masks.float()
        loss = loss.mean()

        return loss

class PoseNet(nn.Module):
    def __init__(self, inp_dim, out_dim, hg_dim=256, bn=False):
        super(PoseNet, self).__init__()
        self.pre = PosePre(img_dim=inp_dim, hg_dim=hg_dim, bn=bn)
        self.hgmod = PoseHGModule(inp_dim=hg_dim, out_dim=out_dim, merge=True, bn=bn)

    def forward(self, x, merge=None):
        if merge is not None:
            return self.hgmod(self.pre(x) + merge)
        else:
            return self.hgmod(self.pre(x))
