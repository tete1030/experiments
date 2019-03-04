#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

void displace_pos_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out);

void displace_pos_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    at::Tensor data_out);

void displace_pos_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in, at::Tensor grad_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_pos_backward_cuda(
    cudaStream_t stream,
    const at::optional<at::Tensor> data_in, at::Tensor grad_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    at::optional<at::Tensor> grad_offsets_x,
    at::optional<at::Tensor> grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_pos_backward_data_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_pos_backward_data_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_pos_backward_offset_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void displace_pos_backward_offset_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    at::Tensor grad_offsets_x,
    at::Tensor grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);
