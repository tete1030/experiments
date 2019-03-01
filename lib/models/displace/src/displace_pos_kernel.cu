#include "kernel.cuh"
#include "displace_pos_kernel.h"

#define OFFSET_AS_VECTOR

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_forward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets, const int64_t channel_per_offset,
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
    int64_t num_offset = num_channel / channel_per_offset;
#ifdef OFFSET_AS_VECTOR
    offsets += (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out) * 2;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + 1);
#else
    offsets += ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + num_offset * height_out * width_out);
#endif
    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    data_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;
    if (h_in_int >= 0 && h_in_int < height_in) {
        if (w_in_int >= 0 && w_in_int < width_in) {
            tl = *data_in;
        }
        if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
            tr = *(data_in + 1);
        }
    }
    if (h_in_int + 1 >= 0 && h_in_int + 1 < height_in) {
        if (w_in_int >= 0 && w_in_int < width_in) {
            bl = *(data_in + width_in);
        }
        if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
            br = *(data_in + width_in + 1);
        }
    }
    *data_out = tl * (1 - w_in) * (1 - h_in) + tr * w_in * (1 - h_in) + 
                bl * (1 - w_in) * h_in       + br * w_in * h_in;
  }
}

void displace_pos_forward_cuda(
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
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_forward_cuda", ([&] {
    displace_pos_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), height_in, width_in,
      offsets.data<float>(), channel_per_offset,
      data_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_forward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y, const int64_t channel_per_offset,
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
    int64_t num_offset = num_channel / channel_per_offset;

    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out);
    offsets_x += offset_index;
    offsets_y += offset_index;
    float w_in = w_out - *offsets_x;
    float h_in = h_out - *offsets_y;

    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    data_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;
    if (h_in_int >= 0 && h_in_int < height_in) {
        if (w_in_int >= 0 && w_in_int < width_in) {
            tl = *data_in;
        }
        if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
            tr = *(data_in + 1);
        }
    }
    if (h_in_int + 1 >= 0 && h_in_int + 1 < height_in) {
        if (w_in_int >= 0 && w_in_int < width_in) {
            bl = *(data_in + width_in);
        }
        if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
            br = *(data_in + width_in + 1);
        }
    }
    *data_out = tl * (1 - w_in) * (1 - h_in) + tr * w_in * (1 - h_in) + 
                bl * (1 - w_in) * h_in       + br * w_in * h_in;
  }
}

void displace_pos_forward_cuda(
  cudaStream_t stream,
  const at::Tensor data_in,
  const at::Tensor offsets_x,
  const at::Tensor offsets_y,
  const int64_t channel_per_offset,
  at::Tensor data_out) {
int64_t batch_size = data_in.size(0);
int64_t num_channel = data_in.size(1);
int64_t height_in = data_in.size(2);
int64_t width_in = data_in.size(3);
int64_t height_out = data_out.size(2);
int64_t width_out = data_out.size(3);
int64_t num_kernel = batch_size * num_channel * height_out * width_out;
AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_forward_cuda", ([&] {
  displace_pos_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
    num_kernel, num_channel,
    data_in.data<scalar_t>(), height_in, width_in,
    offsets_x.data<float>(), offsets_y.data<float>(), channel_per_offset,
    data_out.data<scalar_t>(), height_out, width_out);
}));
gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets, float* __restrict__ grad_offsets, const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;
#ifdef OFFSET_AS_VECTOR
    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out) * 2;
    offsets += offset_index;
    grad_offsets += offset_index;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + 1);
#else
    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets += offset_index;
    grad_offsets += offset_index;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + num_offset * height_out * width_out);
#endif
    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    int64_t data_in_index = ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;
    data_in += data_in_index;
    grad_in += data_in_index;

    // if (h_in_int >= 0 && h_in_int < height_in) {
    //     if (w_in_int >= 0 && w_in_int < width_in) {
    //         atomicAdd(grad_in, *grad_out * (1-w_in) * (1-h_in));
    //     }
    //     if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
    //         atomicAdd(grad_in + 1, *grad_out * w_in * (1-h_in));
    //     }
    // }
    // if (h_in_int + 1 >= 0 && h_in_int + 1 < height_in) {
    //     if (w_in_int >= 0 && w_in_int < width_in) {
    //         atomicAdd(grad_in + width_in, *grad_out * (1-w_in) * h_in);
    //     }
    //     if (w_in_int + 1 >= 0 && w_in_int + 1 < width_in) {
    //         atomicAdd(grad_in + width_in + 1, *grad_out * w_in * h_in);
    //     }
    // }

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        atomicAdd(grad_in, *grad_out * (1-w_in) * (1-h_in));
        atomicAdd(grad_in + 1, *grad_out * w_in * (1-h_in));
        atomicAdd(grad_in + width_in, *grad_out * (1-w_in) * h_in);
        atomicAdd(grad_in + width_in + 1, *grad_out * w_in * h_in);


        tl = *data_in;
        tr = *(data_in + 1);
        bl = *(data_in + width_in);
        br = *(data_in + width_in + 1);
        // Only calculate grad when all inside (since edges will cause grad inaccuracy)
        atomicAdd(grad_offsets, - *grad_out * ((tr - tl) * (1 - h_in) + (br - bl) * h_in));
        atomicAdd(
#ifdef OFFSET_AS_VECTOR
          grad_offsets + 1,
#else
          grad_offsets + num_offset * height_out * width_out,
#endif
          - *grad_out * ((bl - tl) * (1 - w_in) + (br - tr) * w_in));
    }
    
  }
}

void displace_pos_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in, at::Tensor grad_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_backward_cuda", ([&] {
    displace_pos_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), grad_in.data<scalar_t>(),
      height_in, width_in,
      offsets.data<float>(), grad_offsets.data<float>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y,
    float* __restrict__ grad_offsets_x, float* __restrict__ grad_offsets_y,
    const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;

    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out);
    offsets_x += offset_index;
    offsets_y += offset_index;
    grad_offsets_x += offset_index;
    grad_offsets_y += offset_index;
    float w_in = w_out - *offsets_x;
    float h_in = h_out - *offsets_y;

    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    int64_t data_in_index = ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;
    data_in += data_in_index;
    grad_in += data_in_index;

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        atomicAdd(grad_in, *grad_out * (1-w_in) * (1-h_in));
        atomicAdd(grad_in + 1, *grad_out * w_in * (1-h_in));
        atomicAdd(grad_in + width_in, *grad_out * (1-w_in) * h_in);
        atomicAdd(grad_in + width_in + 1, *grad_out * w_in * h_in);

        tl = *data_in;
        tr = *(data_in + 1);
        bl = *(data_in + width_in);
        br = *(data_in + width_in + 1);
        // Only calculate grad when all inside (since edges will cause grad inaccuracy)
        atomicAdd(grad_offsets_x, - *grad_out * ((tr - tl) * (1 - h_in) + (br - bl) * h_in));
        atomicAdd(grad_offsets_y, - *grad_out * ((bl - tl) * (1 - w_in) + (br - tr) * w_in));
    }
    
  }
}

void displace_pos_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in, at::Tensor grad_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    at::Tensor grad_offsets_x, at::Tensor grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_backward_cuda", ([&] {
    displace_pos_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), grad_in.data<scalar_t>(),
      height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(),
      grad_offsets_x.data<float>(), grad_offsets_y.data<float>(),
      channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_data_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets, const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;
#ifdef OFFSET_AS_VECTOR
    offsets += (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out) * 2;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + 1);
#else
    offsets += ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + num_offset * height_out * width_out);
#endif
    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    grad_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        atomicAdd(grad_in, *grad_out * (1-w_in) * (1-h_in));
        atomicAdd(grad_in + 1, *grad_out * w_in * (1-h_in));
        atomicAdd(grad_in + width_in, *grad_out * (1-w_in) * h_in);
        atomicAdd(grad_in + width_in + 1, *grad_out * w_in * h_in);
    }
    
  }
}

void displace_pos_backward_data_cuda(
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
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "displace_pos_backward_data_cuda", ([&] {
    displace_pos_backward_data_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      grad_in.data<scalar_t>(),
      height_in, width_in,
      offsets.data<float>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_data_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y, const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;

    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out);
    offsets_x += offset_index;
    offsets_y += offset_index;
    float w_in = w_out - *offsets_x;
    float h_in = h_out - *offsets_y;

    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    grad_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        atomicAdd(grad_in, *grad_out * (1-w_in) * (1-h_in));
        atomicAdd(grad_in + 1, *grad_out * w_in * (1-h_in));
        atomicAdd(grad_in + width_in, *grad_out * (1-w_in) * h_in);
        atomicAdd(grad_in + width_in + 1, *grad_out * w_in * h_in);
    }
    
  }
}

void displace_pos_backward_data_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = grad_in.size(2);
  int64_t width_in = grad_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "displace_pos_backward_data_cuda", ([&] {
    displace_pos_backward_data_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      grad_in.data<scalar_t>(),
      height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_offset_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets, float* __restrict__ grad_offsets, const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;
#ifdef OFFSET_AS_VECTOR
    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out) * 2;
    offsets += offset_index;
    grad_offsets += offset_index;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + 1);
#else
    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets += offset_index;
    grad_offsets += offset_index;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + num_offset * height_out * width_out);
#endif
    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    data_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        tl = *data_in;
        tr = *(data_in + 1);
        bl = *(data_in + width_in);
        br = *(data_in + width_in + 1);
        // Only calculate grad when all inside (since edges will cause grad inaccuracy)
        atomicAdd(grad_offsets, - *grad_out * ((tr - tl) * (1 - h_in) + (br - bl) * h_in));
        atomicAdd(
#ifdef OFFSET_AS_VECTOR
          grad_offsets + 1,
#else
          grad_offsets + num_offset * height_out * width_out,
#endif
          - *grad_out * ((bl - tl) * (1 - w_in) + (br - tr) * w_in));
    }
    
  }
}

void displace_pos_backward_offset_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_backward_offset_cuda", ([&] {
    displace_pos_backward_offset_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(),
      height_in, width_in,
      offsets.data<float>(), grad_offsets.data<float>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_pos_backward_offset_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y,
    float* __restrict__ grad_offsets_x, float* __restrict__ grad_offsets_y,
    const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out) {
  CUDA_KERNEL_LOOP(index, n) {
    grad_out += index;
    int64_t w_out = index % width_out;
    index /= width_out;
    int64_t h_out = index % height_out;
    index /= height_out;
    int64_t i_channel = index % num_channel;
    int64_t i_samp = index / num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    int64_t num_offset = num_channel / channel_per_offset;

    int64_t offset_index = (((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out) * 2;
    offsets_x += offset_index;
    offsets_y += offset_index;
    grad_offsets_x += offset_index;
    grad_offsets_y += offset_index;
    float w_in = w_out - *offsets_x;
    float h_in = h_out - *offsets_y;

    int64_t w_in_int = (int64_t)floorf(w_in);
    int64_t h_in_int = (int64_t)floorf(h_in);
    w_in = w_in - w_in_int;
    h_in = h_in - h_in_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    data_in += ((i_samp * num_channel + i_channel) * height_in + h_in_int) * width_in + w_in_int;

    if (w_in_int >= 0 && w_in_int + 1 < width_in && h_in_int >= 0 && h_in_int + 1 < height_in) {
        tl = *data_in;
        tr = *(data_in + 1);
        bl = *(data_in + width_in);
        br = *(data_in + width_in + 1);
        // Only calculate grad when all inside (since edges will cause grad inaccuracy)
        atomicAdd(grad_offsets_x, - *grad_out * ((tr - tl) * (1 - h_in) + (br - bl) * h_in));
        atomicAdd(grad_offsets_y, - *grad_out * ((bl - tl) * (1 - w_in) + (br - tr) * w_in));
    }
    
  }
}

void displace_pos_backward_offset_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    at::Tensor grad_offsets_x,
    at::Tensor grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_pos_backward_offset_cuda", ([&] {
    displace_pos_backward_offset_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(),
      height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(),
      grad_offsets_x.data<float>(), grad_offsets_y.data<float>(),
      channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}