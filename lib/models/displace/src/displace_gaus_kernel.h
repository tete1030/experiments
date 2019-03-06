#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

void displace_gaus_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    at::Tensor data_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales, const at::Tensor gaus_weight,
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    // dtype
    float fill);

void displace_gaus_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in, at::optional<at::Tensor> grad_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    at::optional<at::Tensor> grad_offsets_x, at::optional<at::Tensor> grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales,
    const at::Tensor gaus_weight, at::optional<at::Tensor> grad_gaus_weight,
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    // dtype
    float fill, bool divide_distance);

void displace_gaus_simple_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    at::optional<at::Tensor> grad_offsets_x, at::optional<at::Tensor> grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales,
    const at::Tensor gaus_weight, at::optional<at::Tensor> grad_gaus_weight,
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    // dtype
    float fill, bool divide_distance);