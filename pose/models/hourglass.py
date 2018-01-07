'''
Hourglass network inserted in the pre-activated Resnet 
Use lr=0.01 for current version
(c) YANG, Wei 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'MaskHourglassNet', 'PathHourglassNet', 'HGDynamicWeightMSELoss', 'Bottleneck', 'hg']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()
        assert type(num_classes) is int or (type(num_classes) is list and len(num_classes) == num_stacks)
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            cur_classes = num_classes if type(num_classes) is int else num_classes[i]
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, cur_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(cur_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

class MaskHourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16, mask=False):
        super(MaskHourglassNet, self).__init__()
        assert type(num_classes) is int or (type(num_classes) is list and len(num_classes) == num_stacks)
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            cur_classes = num_classes if type(num_classes) is int else num_classes[i]
            if mask:
                cur_classes *= 2
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, cur_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(cur_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_) 
        self.score_ = nn.ModuleList(score_)
        self.sigmoid = nn.Sigmoid()
        self.mask = mask
        self.num_classes = num_classes

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        for i in range(self.num_stacks):
            cur_classes = self.num_classes if type(self.num_classes) is int else self.num_classes[i]

            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            # QT: added relu for mask multiplying
            score = self.score[i](y)
            if self.mask:
                score[:, cur_classes:] = self.sigmoid(score[:, cur_classes:])
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out

class PathHourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(PathHourglassNet, self).__init__()
        assert type(num_classes) is int or (type(num_classes) is list and len(num_classes) == num_stacks)
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, score_, score_res, start_res = [], [], [], [], [], [], []
        for i in range(num_stacks):
            cur_classes = num_classes if type(num_classes) is int else num_classes[i]
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, cur_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                score_.append(nn.Conv2d(cur_classes, ch, kernel_size=1, bias=True))
                score_res.append(self._make_residual(block, self.num_feats, num_blocks))
                start_res.append(self._make_residual(block, self.num_feats, num_blocks))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.score_ = nn.ModuleList(score_)
        self.score_res = nn.ModuleList(score_res)
        self.start_res = nn.ModuleList(start_res)
        self.num_classes = num_classes

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def _enhance_score(self, score, level):
        score_max = score.max(-1)[0].max(-1)[0]
        if level == 0:
            k = 3
        elif level == 1:
            k = 8
        topk, topk_idx = score_max.topk(k, dim=1, sorted=True)
        topk = topk.data.cpu()
        topk_idx = topk_idx.data.cpu()
        out_limit = ((topk < 0.2) | (topk > 1.))
        topk[out_limit] = 1.

        eh_scale = torch.ones(score.size()[:2])
        for ehs, tkidx, tk in zip(eh_scale, topk_idx, topk):
            ehs[tkidx] = 1. / tk
        #print(topk_idx)
        #print(topk)
        #print(eh_scale)

        eh_scale = torch.autograd.Variable(eh_scale.cuda(), requires_grad=False)
        newscore = score * eh_scale.view(eh_scale.size() + (1, 1))
        
        return newscore

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        start_x = x

        for i in range(self.num_stacks):
            cur_classes = self.num_classes if type(self.num_classes) is int else self.num_classes[i]

            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                score_eh = self._enhance_score(score, i)
                start_ = self.start_res[i](start_x)
                score_ = self.score_res[i](self.score_[i](score_eh))
                # x = x + fc_ + score_
                x = start_ + score_

        return out

class HGDynamicWeightMSELoss(nn.Module):
    def __init__(self, num_stacks, num_classes, init_weight=None, dynamic=True):
        super(HGDynamicWeightMSELoss, self).__init__()
        if init_weight is None:
            init_weight = torch.zeros((num_classes, num_stacks))
        self.weight = nn.Parameter(init_weight, requires_grad=dynamic)
        self.softmax = nn.Softmax()
    
    def forward(self, input, target):
        actual_weight = self.softmax(self.weight)
        loss_fn = lambda inp, tag, w: (((inp - tag) ** 2).sum(0).sum(-1).sum(-1) * w).sum() / tag.numel()

        loss = loss_fn(input[0], target, actual_weight[:, 0])
        for i in range(1, len(input)):
            loss += loss_fn(input[i], target, actual_weight[:, i])

        return loss

class AssoHourglassNet(nn.Module):
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(AssoHourglassNet, self).__init__()
        assert type(num_classes) is int or (type(num_classes) is list and len(num_classes) == num_stacks)
        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes) 
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, score_, score_res, start_res = [], [], [], [], [], [], []
        for i in range(num_stacks):
            cur_classes = num_classes if type(num_classes) is int else num_classes[i]
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, cur_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                score_.append(nn.Conv2d(cur_classes, ch, kernel_size=1, bias=True))
                score_res.append(self._make_residual(block, self.num_feats, num_blocks))
                start_res.append(self._make_residual(block, self.num_feats, num_blocks))

        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.score_ = nn.ModuleList(score_)
        self.score_res = nn.ModuleList(score_res)
        self.start_res = nn.ModuleList(start_res)
        self.num_classes = num_classes

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 

        x = self.layer1(x)  
        x = self.maxpool(x)
        x = self.layer2(x)  
        x = self.layer3(x)  

        start_x = x

        for i in range(self.num_stacks):
            cur_classes = self.num_classes if type(self.num_classes) is int else self.num_classes[i]

            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                start_ = self.start_res[i](start_x)
                score_ = self.score_res[i](self.score_[i](score))
                # x = x + fc_ + score_
                x = start_ + score_

        return out


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model
