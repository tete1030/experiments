#include <torch/torch.h>
#include <THC/THC.h>
#include <THC/THCGeneral.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void displace_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out);

void displace_backward_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Int, "dtype of offsets must be int");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_forward_cuda(stream, data_in, offsets, channel_per_offset, data_out);
}

void displace_backward(
    const int64_t state,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Int, "dtype of offsets must be int");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_backward_cuda(stream, grad_in, offsets, channel_per_offset, grad_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("displace_forward", &displace_forward, "displace forward");
  m.def("displace_backward", &displace_backward, "displace backward");
}
