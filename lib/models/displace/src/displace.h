#ifndef _DISPLACE_H
#define _DISPLACE_H

#include <torch/extension.h>
#include <THC/THC.h>
#include <THC/THCGeneral.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include "displace_kernel.h"
#include "displace_pos_kernel.h"
#include "displace_gaus_kernel.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#endif