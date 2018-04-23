import torch
import os
import time
from torch.autograd import Function
from torch import nn
import importlib

import lib as _lib

class AELossFunction(Function):
    def forward(self, tags, keypoints):
        # tags: #batch x (17*h*w) x 1
        # inter-person loss, intra-person loss
        output = torch.zeros(torch.Size((tags.size()[0], 2)))
        # #batch x #batch_i_person x (tag_dim + 1(store num_joints))
        mean_tags = torch.zeros(torch.Size((tags.size()[0], keypoints.size()[1], tags.size()[2]+1)))
        self.mean_tags = mean_tags

        _lib.ae_loss_forward(tags, keypoints, output, mean_tags)
        self.save_for_backward(tags, keypoints)
        return output

    def backward(self, grad_output):
        tags, keypoints = self.saved_tensors
        grad_input = torch.zeros(tags.size()).cuda(tags.get_device())
        #grad_input = tags.new(tags.size()).zero_()
        _lib.ae_loss_backward(tags, keypoints, self.mean_tags, grad_output, grad_input)
        self.mean_tags = None
        return grad_input, torch.zeros(keypoints.size())

class AELoss(nn.Module):
    def forward(self, input, input1):
        if not input.is_cuda:
            input = input.cuda()
        output = AELossFunction()(input, input1)
        return output
