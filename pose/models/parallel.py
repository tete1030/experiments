##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Data Parallel"""
import threading
import torch
from torch.autograd import Function
import torch.cuda.comm as comm
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Broadcast

__all__ = ['DataParallelModel', 'DataParallelCriterion']

class ReduceAdd(Function):
    @staticmethod
    def forward(ctx, destination, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        return comm.reduce_add(inputs, destination)

    @staticmethod
    def backward(ctx, gradOutput):
        return (None,) + Broadcast.apply(ctx.target_gpus, gradOutput)

def sum_output(outputs, target_device, comp_mean=False):
    r"""
    Sum tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def sum_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.autograd.Variable):
            result = ReduceAdd.apply(target_device, *outputs)
            if comp_mean:
                result /= len(outputs)
            return result
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, sum_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(sum_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return sum_map(outputs)
    finally:
        sum_map = None

class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. "Context Encoding for Semantic Segmentation".
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def gather(self, outputs, output_device):
        return outputs


class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. "Context Encoding for Semantic Segmentation".
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """

    def __init__(self, *args, **kwargs):
        self.comp_mean = kwargs.pop('comp_mean', True)
        super(DataParallelCriterion, self).__init__(*args, **kwargs)

    def forward(self, scattered_inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        if not self.device_ids:
            return self.module(scattered_inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(scattered_inputs, *targets[0], **kwargs[0])
        inputs = map(lambda a, seq: (a,) + tuple(seq), scattered_inputs, targets)
        replicas = replicate(self.module, self.device_ids[:len(inputs)])
        outputs = parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(inputs)])
        return sum_output(outputs, self.output_device, comp_mean=self.comp_mean)
