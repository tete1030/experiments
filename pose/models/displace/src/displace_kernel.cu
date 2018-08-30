#include "ATen/ATen.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

inline int GET_BLOCKS(const int N)
{
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_forward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, const int64_t height_in, const int64_t width_in,
    const int* __restrict__ offsets, const int64_t channel_per_offset,
    Dtype* __restrict__ data_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    data_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t w_in = w_out - offsets[i_offset * 2];
    int64_t h_in = h_out - offsets[i_offset * 2 + 1];

    if (w_in >= 0 && h_in >= 0 && w_in < width_in && h_in < height_in) {
        data_in += ((i_samp * num_channel + i_channel) * height_in + h_in) * width_in + w_in;
        *data_out = *data_in;
    } else {
        *data_out = 0;
    }
  }
}

void displace_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out) {
  int64_t batch_size = data_in.size(0);
  int64_t num_channel = data_in.size(1);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t height_out = data_out.size(2);
  int64_t width_out = data_out.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_forward_cuda", ([&] {
    displace_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), height_in, width_in,
      offsets.data<int>(), channel_per_offset,
      data_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}


template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_backward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    Dtype* __restrict__ grad_in, const int64_t height_in, const int64_t width_in,
    const int* __restrict__ offsets, const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_in += index;
    int64_t w_in = index % width_in;
    index /= width_in;
    int64_t h_in = index % height_in;
    index /= height_in;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t w_out = w_in + offsets[i_offset * 2];
    int64_t h_out = h_in + offsets[i_offset * 2 + 1];

    if (w_out >= 0 && h_out >= 0 && w_out < width_out && h_out < height_out) {
        grad_out += ((i_samp * num_channel + i_channel) * height_out + h_out) * width_out + w_out;
        *grad_in = *grad_out;
    } else {
        *grad_in = 0;
    }
  }
}

void displace_backward_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = grad_in.size(2);
  int64_t width_in = grad_in.size(3);
  int64_t num_kernel = num_channel * height_in * width_in;
  
  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "displace_backward_cuda", ([&] {
    displace_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      grad_in.data<scalar_t>(), height_in, width_in,
      offsets.data<int>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}