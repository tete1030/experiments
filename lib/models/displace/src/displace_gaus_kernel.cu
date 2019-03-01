#include "kernel.cuh"
#include "displace_gaus_kernel.h"
#include <math_constants.h>

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_gaus_forward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y, const int64_t channel_per_offset,
    Dtype* __restrict__ data_out, const int64_t height_out, const int64_t width_out,
    const float* __restrict__ gaus_angles, const float* __restrict__ gaus_scales, const Dtype* __restrict__ gaus_weight,
    const int64_t num_gaus, float fill) {
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

    gaus_angles += i_offset * num_gaus;
    gaus_scales += i_offset * num_gaus;
    gaus_weight += i_offset * num_gaus;

    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets_x += offset_index;
    offsets_y += offset_index;

    Dtype val_out = 0.;
    float val_offset_x = *offsets_x;
    float val_offset_y = *offsets_y;
    float val_offset_scale = hypotf(val_offset_x, val_offset_y);
    for (int64_t i_gau = 0; i_gau < num_gaus; i_gau++) {
      float gaus_angle = gaus_angles[i_gau];
      float gaus_scale = gaus_scales[i_gau];
      float gaus_scale_ratio = 1 + gaus_scale / val_offset_scale;
      // if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || gaus_scale_ratio < 0) {
      //   continue;
      // }
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F) {
        continue;
      }
      float gaus_angle_cos = cosf(gaus_angle), gaus_angle_sin = sinf(gaus_angle);
      float new_offset_x = (gaus_angle_cos * val_offset_x - gaus_angle_sin * val_offset_y) * gaus_scale_ratio;
      float new_offset_y = (gaus_angle_sin * val_offset_x + gaus_angle_cos * val_offset_y) * gaus_scale_ratio;

      int64_t w_in = w_out - lrintf(new_offset_x);
      int64_t h_in = h_out - lrintf(new_offset_y);
      // float density_scale = abs((val_offset_scale + gaus_scale) / val_offset_scale);
      // float density_scale = hypotf(rintf(new_offset_x), rintf(new_offset_y)) / max(val_offset_scale, 1.);
      // float density_scale = abs(val_offset_scale + gaus_scale) / max(val_offset_scale, 1.);
      if (h_in >= 0 && h_in < height_in) {
        if (w_in >= 0 && w_in < width_in) {
          val_out += data_in[((i_samp * num_channel + i_channel) * height_in + h_in) * width_in + w_in] * gaus_weight[i_gau];
        } else {
          val_out += fill * gaus_weight[i_gau];
        }
      }
    }

    *data_out = val_out;
  }
}

void displace_gaus_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    at::Tensor data_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales, const at::Tensor gaus_weight,
    // dtype
    float fill) {
  int64_t batch_size = data_in.size(0);
  int64_t num_channel = data_in.size(1);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t height_out = data_out.size(2);
  int64_t width_out = data_out.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_gaus_forward_cuda", ([&] {
    displace_gaus_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(), channel_per_offset,
      data_out.data<scalar_t>(), height_out, width_out,
      gaus_angles.data<float>(), gaus_scales.data<float>(), gaus_weight.data<scalar_t>(),
      gaus_weight.size(1), fill);
  }));
  gpuErrchk(cudaGetLastError());
}

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_gaus_backward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, Dtype* __restrict__ grad_in,
    const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y,
    float* __restrict__ grad_offsets_x, float* __restrict__ grad_offsets_y,
    const int64_t channel_per_offset,
    const Dtype* __restrict__ grad_out, const int64_t height_out, const int64_t width_out,
    const float* __restrict__ gaus_angles, const float* __restrict__ gaus_scales,
    const Dtype* __restrict__ gaus_weight, Dtype* __restrict__ grad_gaus_weight,
    const int64_t num_gaus, float fill) {
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

    int64_t gaus_index = i_offset * num_gaus;
    gaus_angles += gaus_index;
    gaus_scales += gaus_index;
    gaus_weight += gaus_index;
    grad_gaus_weight += gaus_index;

    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets_x += offset_index;
    offsets_y += offset_index;
    grad_offsets_x += offset_index;
    grad_offsets_y += offset_index;

    float val_offset_x = *offsets_x;
    float val_offset_y = *offsets_y;
    float val_offset_scale = hypotf(val_offset_x, val_offset_y);
    if (val_offset_scale == 0) {
      val_offset_y = 1e-5;
      val_offset_scale = 1e-5;
    }

    float grad_y = 0, grad_x = 0;
    bool all_in = true;
    for (int64_t i_gau = 0; i_gau < num_gaus; i_gau++) {
      float gaus_angle = gaus_angles[i_gau];
      float gaus_scale = gaus_scales[i_gau];
      float gaus_scale_ratio = 1 + gaus_scale / val_offset_scale;
      // if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || gaus_scale_ratio < 0) {
      //   continue;
      // }
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F) {
        continue;
      }
      float gaus_angle_cos = cosf(gaus_angle), gaus_angle_sin = sinf(gaus_angle);
      float new_offset_x = rint((gaus_angle_cos * val_offset_x - gaus_angle_sin * val_offset_y) * gaus_scale_ratio);
      float new_offset_y = rint((gaus_angle_sin * val_offset_x + gaus_angle_cos * val_offset_y) * gaus_scale_ratio);

      int64_t w_in = w_out - lrintf(new_offset_x);
      int64_t h_in = h_out - lrintf(new_offset_y);
      bool inside = (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in);

      Dtype cur_grad_gaus_weight = 0.;
      Dtype val_gaus_weight = gaus_weight[i_gau];
      Dtype grad_out_scaled = *grad_out * abs(val_offset_scale + gaus_scale) / max(val_offset_scale, 1.);

      if (inside) {
        int64_t in_index = ((i_samp * num_channel + i_channel) * height_in + h_in) * width_in + w_in;
        atomicAdd(
          grad_in + in_index,
          grad_out_scaled * val_gaus_weight
        );
        cur_grad_gaus_weight = grad_out_scaled * data_in[in_index];
      } else {
        all_in = false;
        cur_grad_gaus_weight = grad_out_scaled * fill;
      }
      // ***temp: exclude data_in
      // cur_grad_gaus_weight = *grad_out;

      atomicAdd(
        grad_gaus_weight + i_gau,
        cur_grad_gaus_weight
      );

      // // Use offset different
      // float offoff_x = new_offset_x - val_offset_x;
      // float offoff_y = new_offset_y - val_offset_y;
      // float anglescale_norm = hypot(gaus_angle * val_offset_scale, gaus_scale);
      // float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight * anglescale_norm / hypot(offoff_x, offoff_y);
      // grad_x += cur_grad_off_base * offoff_x;
      // grad_y += cur_grad_off_base * offoff_y;

      // Use angle and scale
      float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight;
      grad_x += cur_grad_off_base *
        (gaus_scale * val_offset_x / val_offset_scale + gaus_angle * (-val_offset_y));
      grad_y += cur_grad_off_base *
        (gaus_scale * val_offset_y / val_offset_scale + gaus_angle * val_offset_x);

      // // Use scale-only
      // float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight;
      // grad_x += cur_grad_off_base *
      //   (gaus_scale * val_offset_x / val_offset_scale);
      // grad_y += cur_grad_off_base *
      //   (gaus_scale * val_offset_y / val_offset_scale);

      // // Use angle-only
      // float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight;
      // grad_x += cur_grad_off_base *
      //   (gaus_angle * (-val_offset_y));
      // grad_y += cur_grad_off_base *
      //   (gaus_angle * val_offset_x);

    }

    atomicAdd(grad_offsets_x, grad_x);
    atomicAdd(grad_offsets_y, grad_y);
    
  }
}

void displace_gaus_backward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in, at::Tensor grad_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    at::Tensor grad_offsets_x, at::Tensor grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales,
    const at::Tensor gaus_weight, at::Tensor grad_gaus_weight,
    // dtype
    float fill) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
  
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_gaus_backward_cuda", ([&] {
    displace_gaus_backward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), grad_in.data<scalar_t>(),
      height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(), grad_offsets_x.data<float>(), grad_offsets_y.data<float>(),
      channel_per_offset,
      grad_out.data<scalar_t>(), height_out, width_out,
      gaus_angles.data<float>(), gaus_scales.data<float>(),
      gaus_weight.data<scalar_t>(), grad_gaus_weight.data<scalar_t>(),
      gaus_weight.size(1), fill);
  }));
  gpuErrchk(cudaGetLastError());
}
