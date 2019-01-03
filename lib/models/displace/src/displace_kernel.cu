#include "kernel.cuh"

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
  int64_t num_kernel = batch_size * num_channel * height_in * width_in;
  
  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "displace_backward_cuda", ([&] {
    displace_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      grad_in.data<scalar_t>(), height_in, width_in,
      offsets.data<int>(), channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_frac_forward_cuda_kernel(
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
    float w_in = w_out - offsets[i_offset * 2];
    float h_in = h_out - offsets[i_offset * 2 + 1];
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

void displace_frac_forward_cuda(
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
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_frac_forward_cuda", ([&] {
    displace_frac_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), height_in, width_in,
      offsets.data<float>(), channel_per_offset,
      data_out.data<scalar_t>(), height_out, width_out);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_frac_backward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets, const int64_t channel_per_offset,
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
    offsets += i_offset * 2;
    float w_out = w_in + *offsets;
    float h_out = h_in + *(offsets + 1);
    int64_t w_out_int = (int64_t)floorf(w_out);
    int64_t h_out_int = (int64_t)floorf(h_out);
    w_out -= w_out_int;
    h_out -= h_out_int;

    Dtype tl=0, tr=0, bl=0, br=0;
    grad_out += ((i_samp * num_channel + i_channel) * height_out + h_out_int) * width_out + w_out_int;

    if (h_out_int >= 0 && h_out_int < height_out) {
        if (w_out_int >= 0 && w_out_int < width_out) {
            tl = *grad_out;
        }
        if (w_out_int + 1 >= 0 && w_out_int + 1 < width_out) {
            tr = *(grad_out + 1);
        }
    }
    if (h_out_int + 1 >= 0 && h_out_int + 1 < height_out) {
        if (w_out_int >= 0 && w_out_int < width_out) {
            bl = *(grad_out + width_out);
        }
        if (w_out_int + 1 >= 0 && w_out_int + 1 < width_out) {
            br = *(grad_out + width_out + 1);
        }
    }
    *grad_in = tl * (1 - w_out) * (1 - h_out) + tr * w_out * (1 - h_out) + 
               bl * (1 - w_out) * h_out       + br * w_out * h_out;
  }
}

void displace_frac_backward_cuda(
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
  int64_t num_kernel = batch_size * num_channel * height_in * width_in;
  
  AT_DISPATCH_FLOATING_TYPES(grad_in.type(), "displace_frac_backward_cuda", ([&] {
    displace_frac_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
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
__global__ void displace_frac_offset_backward_cuda_kernel(
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
    offsets += i_offset * 2;
    grad_offsets += i_offset * 2;
    float w_in = w_out - *offsets;
    float h_in = h_out - *(offsets + 1);
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
        atomicAdd(grad_offsets + 1, - *grad_out * ((bl - tl) * (1 - w_in) + (br - tr) * w_in));
    }
    
  }
}

void displace_frac_offset_backward_cuda(
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
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_frac_offset_backward_cuda", ([&] {
    displace_frac_offset_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
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
__global__ void offset_mask_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ input, const int64_t height, const int64_t width,
    const int* __restrict__ offsets, const int64_t channel_per_offset,
    Dtype* __restrict__ output, const int64_t side_thickness_h, const int64_t side_thickness_w) {
  CUDA_KERNEL_LOOP(index, n) {
    input += index;
    output += index;
    int64_t w = index % width;
    index /= width;
    int64_t h = index % height;
    index /= height;
    int64_t i_channel = index % num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    if (w >= offsets[i_offset * 2] + side_thickness_w && h >= offsets[i_offset * 2 + 1] + side_thickness_h && w < width + offsets[i_offset * 2] - side_thickness_w && h < height + offsets[i_offset * 2 + 1] - side_thickness_h) {
        *output = *input;
    } else {
        *output = 0;
    }
  }
}

void offset_mask_cuda(
    cudaStream_t stream,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output,
    const at::IntList side_thickness) {
  int64_t batch_size = input.size(0);
  int64_t num_channel = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t num_kernel = batch_size * num_channel * height * width;
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "offset_mask_cuda", ([&] {
    offset_mask_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      input.data<scalar_t>(), height, width,
      offsets.data<int>(), channel_per_offset,
      output.data<scalar_t>(), side_thickness[0], side_thickness[1]);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void offset_mask_frac_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ input, const int64_t height, const int64_t width,
    const float* __restrict__ offsets, const int64_t channel_per_offset,
    Dtype* __restrict__ output) {
  CUDA_KERNEL_LOOP(index, n) {
    input += index;
    output += index;
    int64_t w = index % width;
    index /= width;
    int64_t h = index % height;
    index /= height;
    int64_t i_channel = index % num_channel;
    int64_t i_offset = i_channel / channel_per_offset;
    offsets += i_offset * 2;
    if (w >= *offsets && h >= *(offsets+1) && w <= width - 1 + *offsets && h <= height - 1 + *(offsets+1)) {
        *output = *input;
    } else {
        *output = 0;
    }
  }
}

void offset_mask_frac_cuda(
    cudaStream_t stream,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output) {
  int64_t batch_size = input.size(0);
  int64_t num_channel = input.size(1);
  int64_t height = input.size(2);
  int64_t width = input.size(3);
  int64_t num_kernel = batch_size * num_channel * height * width;
  
  AT_DISPATCH_FLOATING_TYPES(input.type(), "offset_mask_frac_cuda", ([&] {
    offset_mask_frac_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      input.data<scalar_t>(), height, width,
      offsets.data<float>(), channel_per_offset,
      output.data<scalar_t>());
  }));
  gpuErrchk(cudaGetLastError());
}