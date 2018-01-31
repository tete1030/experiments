import numpy as np
import torch
from torch import nn
from torch import autograd
from .simphg import Conv, Pool, SimpHourglass

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseImagePre(nn.Module):
    def __init__(self, hg_dim, bn=False):
        super(PoseImagePre, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, hg_dim, bn=bn)
        )

    def forward(self, imgs):
        x = self.pre(imgs)
        return x

class PoseHGModule(nn.Module):
    def __init__(self, inp_dim, oup_dim, merge, bn=False, increase=128):
        super(PoseHGModule, self).__init__()
        self.features = nn.Sequential(
            SimpHourglass(4, inp_dim, bn, increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        )
        self.outs = Conv(inp_dim, oup_dim, 1, relu=False, bn=False)
        if merge:
            self.merge_features = Merge(inp_dim, inp_dim)
            self.merge_preds = Merge(oup_dim, inp_dim)

    def forward(self, x):
        feature = self.features(x)
        pred = self.outs(feature)
        if self.merge_features:
            x = x + self.merge_preds(pred) + self.merge_features(feature)
        return pred, x

class PoseMapGenerator(nn.Module):
    def __init__(self, batch_size, num_parts, out_size, max_num_people, sigma=1):
        # size: batch_size x (num_parts*2) x h x w
        super(PoseMapGenerator, self).__init__()
        self.batch_size = batch_size
        self.num_parts = num_parts
        self.out_size = out_size
        self.max_num_people = max_num_people
        self.sigma = sigma

        self.map = autograd.Variable(torch.zeros(batch_size, num_parts*2, *out_size), requires_grad=False)
        temp_size = 6*sigma + 3
        x = np.arange(0, temp_size, 1, np.float32)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        keypoints_temp = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        embedding_temp = (np.sqrt(((x - x0) ** 2 + (y - y0) ** 2)) <= (3*sigma + 1)).astype(np.float32)
        self.keypoints_temp = autograd.Variable(torch.from_numpy(keypoints_temp * embedding_temp), requires_grad=False)
        self.embedding_temp = autograd.Variable(torch.from_numpy(embedding_temp), requires_grad=False)

    @classmethod
    def init_ids(cls, keypoints):
        # keypoints: #batch x [#batch_i_person x #part x 3]
        num_parts = keypoints[0].size(1)
        split = torch.cumsum(torch.Tensor([k.size(0) for k in keypoints]).long(), dim=0)
        batch_all_ids = torch.IntTensor(split[-1]).long()
        person_all_ids = torch.IntTensor(split[-1]).long()
        if len(split) > 0:
            last_end_pos = 0
            for ib, end_pos in enumerate(split):
                batch_all_ids[last_end_pos:end_pos] = ib
                person_all_ids[last_end_pos:end_pos] = torch.arange(end=end_pos-last_end_pos)
                last_end_pos = end_pos
        else:
            batch_all_ids[:] = 0
            person_all_ids[:] = torch.arange(end=split[-1])

        # keypoints: #all_person x #part x 3
        keypoints = torch.cat(keypoints, dim=0)

        # flat_ids, part_ids: #valid
        flat_ids, part_ids = torch.nonzero(keypoints[:, :, 2] >= 1).t()
        batch_ids = batch_all_ids[flat_ids]
        person_ids = person_all_ids[flat_ids]

        # keypoints_flat: #valid_part x 2
        keypoints_flat = keypoints[flat_ids, part_ids][:, :2].long()

        # THE FOLLOWING CODE IS USED WHEN KEYPOINTS ARE CONSIST OF LIST AND DICT
        # batch_ids = []
        # person_ids = []
        # part_ids = []
        # keypoints_flat = []
        # for i, sample in enumerate(keypoints):
        #     batch_ids += [i] * len(sample)
        #     keypoints_sample = []
        #     for j, person in enumerate(sample):
        #         person_ids += [j] * len(person)
        #         keypoints_person = np.stack(person.values(), axis=0)
        #         keypoints_sample.append(keypoints_person)
        #         for k, part in person.iteritems():
        #             part_ids.append(k)
        #     keypoints_sample = np.concatenate(keypoints_sample)
        #     keypoints_flat.append(keypoints_sample)
        # keypoints_flat = np.concatenate(keypoints_flat)

        return batch_ids, person_ids, part_ids, keypoints_flat

    def forward(self, batch_ids, person_ids, part_ids, keypoints_flat):
        # batch_ids, person_ids, part_ids: #all_parts
        # keypoints_flat: #all_parts x 2
        batch_ids = batch_ids.contiguous()
        person_ids = person_ids.contiguous()
        part_ids = part_ids.contiguous()

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
        insider = ((map_cols >= 0) & (map_cols < self.out_size[1]) & (map_rows >= 0) & (map_rows < self.out_size[0]))
        outsider = ~insider

        # in case out of bound
        batch_ids[outsider] = 0
        part_ids[outsider] = 0
        map_rows[outsider] = 0
        map_cols[outsider] = 0

        # embeddings: #batch x max_num_people
        embeddings = []
        for i in range(max_batch_id+1):
            emb = np.arange(1, self.max_num_people+1)
            np.random.shuffle(emb)
            embeddings.append(emb)
        # embeddings: max_num_people x #batch
        embeddings = autograd.Variable(torch.from_numpy(np.array(embeddings).T).float())

        self.map.zero_()
        
        for i, pim in enumerate(person_id_masks):
            # pim: #all_parts x temp_size
            # select the part that is overlapped with other points but nearer to this point
            # modify_mask: #all_parts x temp_size
            
            modify_mask = torch.lt(self.map[batch_ids, part_ids, map_rows, map_cols], self.keypoints_temp.view(-1)).data
            mask = (pim & modify_mask & insider)

            # #point
            bids_mask = batch_ids[mask]
            pids_mask = part_ids[mask]
            map_row_mask = map_rows[mask]
            map_col_mask = map_cols[mask]
            temp_row_mask = temp_rows[mask]
            temp_col_mask = temp_cols[mask]

            self.map[bids_mask, pids_mask, map_row_mask, map_col_mask] = \
                self.keypoints_temp[temp_row_mask, temp_col_mask]

            self.map[bids_mask, pids_mask+self.num_parts, map_row_mask, map_col_mask] = \
                self.embedding_temp[temp_row_mask, temp_col_mask] * embeddings[i][bids_mask]

        return self.map


class PoseNet(nn.Module):
    def __init__(self, nstack, stack_shared, inp_dim, oup_dim, bn=False, increase=128):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )

        if not stack_shared:
            self.features = nn.ModuleList([
                nn.Sequential(
                    SimpHourglass(4, inp_dim, bn, increase),
                    Conv(inp_dim, inp_dim, 3, bn=False),
                    Conv(inp_dim, inp_dim, 3, bn=False)
                ) for i in range(nstack)] )
            self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
            self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
            self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        else:
            self.features = nn.ModuleList([
                nn.Sequential(
                    SimpHourglass(4, inp_dim, bn, increase),
                    Conv(inp_dim, inp_dim, 3, bn=False),
                    Conv(inp_dim, inp_dim, 3, bn=False)
                )])
            self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False)])
            if self.nstack and self.nstack < 2:
                self.merge_features = None
                self.merge_preds = None
            else:
                self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim)])
                self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim)])

        self.nstack = nstack
        self.stack_shared = stack_shared

    def forward(self, imgs):
        x = self.pre(imgs)
        preds = []
        i = 0

        def stacks_generator():
            if not self.stack_shared:
                for i in range(self.nstack):
                    yield i, i, (i >= self.nstack-1)
            else:
                if self.nstack:
                    for i in range(self.nstack):
                        yield i, 0, (i >= self.nstack-1)
                else:
                    i = 0
                    while True:
                        yield i, 0, False
                        i += 1

        for i, idx, is_end in stacks_generator():
            feature = self.features[idx](x)
            preds.append(self.outs[idx](feature))
            if not is_end:
                x = x + self.merge_preds[idx](preds[-1]) + self.merge_features[idx](feature)
            # TODO
            if accurate():
                break
        # TODO not always stack all preds
        return torch.stack(preds, 1)

