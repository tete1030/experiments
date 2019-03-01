#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

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

void displace_frac_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out);

void displace_frac_backward_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_frac_offset_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void offset_mask_frac_cuda(
    cudaStream_t stream,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output);

void offset_mask_cuda(
    cudaStream_t stream,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output,
    const at::IntList side_thickness);