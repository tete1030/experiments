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
from torch.nn.parallel.scatter_gather import gather

__all__ = ['DataParallelModel', 'DataParallelCriterion', 'gather']

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

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return [self.module(*inputs, **kwargs)]
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return [self.module(*inputs[0], **kwargs[0])]
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
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
        self.store_replicas = kwargs.pop('store_replicas', False)
        super(DataParallelCriterion, self).__init__(*args, **kwargs)
        assert self.device_ids, "empty device_ids is not supported"
        if self.store_replicas:
            self.replicas = self.replicate_module()

    def replicas_each(self, method, shared_args=None, shared_kwargs=None, ind_args=None, ind_kwargs=None):
        assert self.store_replicas and self.replicas
        
        for i, mod in enumerate(self.replicas):
            args = tuple()
            if shared_args:
                args += shared_args
            if ind_args:
                args += ind_args[i]
            kwargs = dict()
            if shared_kwargs:
                kwargs.update(shared_kwargs)
            if ind_kwargs:
                kwargs.update(ind_kwargs[i])
            getattr(mod, method).__call__(*args, **kwargs)

    def replicate_module(self, input_length=None):
        if len(self.device_ids) == 1:
            return [self.module]
        else:
            if input_length:
                return replicate(self.module, self.device_ids[:input_length])
            else:
                return replicate(self.module, self.device_ids)

    def forward(self, scattered_inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        assert self.device_ids, "empty device_ids is not supported"
        assert isinstance(scattered_inputs, list)

        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(scattered_inputs) < len(targets):
            scattered_inputs.extend([() for _ in range(len(targets) - len(scattered_inputs))])
        elif len(targets) < len(scattered_inputs):
            complement_len = len(scattered_inputs) - len(targets)
            targets += tuple([() for _ in range(complement_len)])
            kwargs += tuple([{} for _ in range(complement_len)])
        assert len(scattered_inputs) <= len(self.device_ids)

        inputs = map(lambda a, seq: (a,) + tuple(seq), scattered_inputs, targets)
        if self.store_replicas:
            replicas = self.replicas[:len(inputs)]
        else:
            replicas = self.replicate_module(input_length=len(inputs))

        if len(self.device_ids) == 1:
            return replicas[0](*inputs[0], **kwargs[0])
        else:
            outputs = parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(inputs)])
            return sum_output(outputs, self.output_device, comp_mean=self.comp_mean)
