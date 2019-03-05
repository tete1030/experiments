#include "kernel.cuh"
#include "displace_gaus_kernel.h"
#include <math_constants.h>

// #define SEP_SAMPLE
// #define USE_FORWARD_SIDE_BALANCE
// #define USE_BACKWARD_SIDE_BALANCE

// #define FILTER_ANGLE
// #define FILTER_SCALE
// #define FILTER_BOTH

// #define FILTER_SCALE_BOTH_SIDE

template <typename Dtype>
__launch_bounds__(CUDA_NUM_THREADS)
__global__ void displace_gaus_forward_cuda_kernel(
    const int64_t n, const int64_t num_channel,
    const Dtype* __restrict__ data_in, const int64_t height_in, const int64_t width_in,
    const float* __restrict__ offsets_x, const float* __restrict__ offsets_y, const int64_t channel_per_offset,
    Dtype* __restrict__ data_out, const int64_t height_out, const int64_t width_out,
    const float* __restrict__ gaus_angles, const float* __restrict__ gaus_scales, const Dtype* __restrict__ gaus_weight,
    const float* __restrict__ gaus_cos_angles, const float* __restrict__ gaus_sin_angles,
    const int64_t num_gaus, float fill) {
  CUDA_KERNEL_LOOP(index, n) {
#ifdef SEP_SAMPLE
    int64_t i_gau = index % num_gaus;
    index /= num_gaus;
#endif
    data_out += index;
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
    gaus_cos_angles += gaus_index;
    gaus_sin_angles += gaus_index;
    gaus_scales += gaus_index;
    gaus_weight += gaus_index;

    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets_x += offset_index;
    offsets_y += offset_index;

    Dtype val_out = 0.;
    float val_offset_x = *offsets_x;
    float val_offset_y = *offsets_y;
    float val_offset_scale = hypotf(val_offset_x, val_offset_y);
    if (val_offset_scale == 0) {
      val_offset_y = 1e-5;
      val_offset_scale = 1e-5;
    }
#ifndef SEP_SAMPLE
    for (int64_t i_gau = 0; i_gau < num_gaus; i_gau++) {
#else
    do {
#endif // #ifndef SEP_SAMPLE
      float gaus_angle = gaus_angles[i_gau];
      float gaus_scale = gaus_scales[i_gau];
      float gaus_scale_ratio = 1 + gaus_scale / val_offset_scale;
#ifdef FILTER_BOTH
  #ifdef FILTER_SCALE_BOTH
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || abs(gaus_scale) > val_offset_scale) {
        continue;
      }
  #else
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || gaus_scale_ratio < 0) {
        continue;
      }
  #endif
#elif FILTER_ANGLE
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F) {
        continue;
      }
#elif FILTER_SCALE
  #ifdef FILTER_SCALE_BOTH
      if (abs(gaus_scale) > val_offset_scale) {
        continue;
      }
  #else
      if (gaus_scale_ratio < 0) {
        continue;
      }
  #endif
#endif

      float gaus_angle_cos = gaus_cos_angles[i_gau], gaus_angle_sin = gaus_sin_angles[i_gau];
      float new_offset_x = (gaus_angle_cos * val_offset_x - gaus_angle_sin * val_offset_y) * gaus_scale_ratio;
      float new_offset_y = (gaus_angle_sin * val_offset_x + gaus_angle_cos * val_offset_y) * gaus_scale_ratio;

      int64_t w_in = w_out - lrintf(new_offset_x);
      int64_t h_in = h_out - lrintf(new_offset_y);
      bool inside = (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in);
      // float density_scale = abs((val_offset_scale + gaus_scale) / val_offset_scale);
      // float density_scale = hypotf(rintf(new_offset_x), rintf(new_offset_y)) / max(val_offset_scale, 1.);
      // float density_scale = abs(val_offset_scale + gaus_scale) / max(val_offset_scale, 1.);
#ifdef USE_FORWARD_SIDE_BALANCE
      Dtype gaus_weight_density = gaus_weight[i_gau] * abs(val_offset_scale + gaus_scale) / max(val_offset_scale, 1.0);
#else
      Dtype gaus_weight_density = gaus_weight[i_gau];
#endif
      if (inside) {
        val_out += data_in[((i_samp * num_channel + i_channel) * height_in + h_in) * width_in + w_in] * gaus_weight_density;
      } else {
        val_out += fill * gaus_weight_density;
      }
#ifndef SEP_SAMPLE
    }
    *data_out = val_out;
#else
    } while (0);
    atomicAdd(data_out, val_out);
#endif

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
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    // dtype
    float fill) {
  int64_t batch_size = data_in.size(0);
  int64_t num_channel = data_in.size(1);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
  int64_t height_out = data_out.size(2);
  int64_t width_out = data_out.size(3);
#ifndef SEP_SAMPLE
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
#else
  int64_t num_kernel = batch_size * num_channel * height_out * width_out * gaus_weight.size(1);
#endif
  AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_gaus_forward_cuda", ([&] {
    displace_gaus_forward_cuda_kernel <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernel, num_channel,
      data_in.data<scalar_t>(), height_in, width_in,
      offsets_x.data<float>(), offsets_y.data<float>(), channel_per_offset,
      data_out.data<scalar_t>(), height_out, width_out,
      gaus_angles.data<float>(), gaus_scales.data<float>(), gaus_weight.data<scalar_t>(),
      gaus_cos_angles.data<float>(), gaus_sin_angles.data<float>(), 
      gaus_weight.size(1), fill);
  }));
  gpuErrchk(cudaGetLastError());
}

template <bool UseGradIn, bool UseGradWeight, bool UseGradOffsets, bool MinusCenter, typename Dtype>
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
    const float* __restrict__ gaus_cos_angles, const float* __restrict__ gaus_sin_angles,
    const int64_t num_gaus, float fill) {
  CUDA_KERNEL_LOOP(index, n) {
#ifdef SEP_SAMPLE
    int64_t i_gau = index % num_gaus;
    index /= num_gaus;
#endif
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
    if (UseGradWeight) {
      grad_gaus_weight += gaus_index;
    }
    gaus_cos_angles += gaus_index;
    gaus_sin_angles += gaus_index;

    int64_t offset_index = ((i_samp * num_offset + i_offset) * height_out + h_out) * width_out + w_out;
    offsets_x += offset_index;
    offsets_y += offset_index;
    if (UseGradOffsets) {
      grad_offsets_x += offset_index;
      grad_offsets_y += offset_index;
    }

    float val_offset_x = *offsets_x;
    float val_offset_y = *offsets_y;
    float val_offset_scale = hypotf(val_offset_x, val_offset_y);
    if (val_offset_scale == 0) {
      val_offset_y = 1e-5;
      val_offset_scale = 1e-5;
    }

    float center_value = 0.;
    if (MinusCenter) {
      int64_t w_in_center = w_out - lrintf(val_offset_x);
      int64_t h_in_center = h_out - lrintf(val_offset_y);
      if (h_in_center >= 0 && h_in_center < height_in && w_in_center >= 0 && w_in_center < width_in) {
        int64_t in_center_index = ((i_samp * num_channel + i_channel) * height_in + h_in_center) * width_in + w_in_center;
        center_value = data_in[in_center_index];
      } else {
        center_value = fill;
      }
    }

    float grad_y = 0, grad_x = 0;
#ifndef SEP_SAMPLE
    for (int64_t i_gau = 0; i_gau < num_gaus; i_gau++) {
#else
    do {
#endif // #ifndef SEP_SAMPLE
      float gaus_angle = gaus_angles[i_gau];
      float gaus_scale = gaus_scales[i_gau];
      float gaus_scale_ratio = 1 + gaus_scale / val_offset_scale;
#ifdef FILTER_BOTH
  #ifdef FILTER_SCALE_BOTH
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || abs(gaus_scale) > val_offset_scale) {
        continue;
      }
  #else
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F || gaus_scale_ratio < 0) {
        continue;
      }
  #endif
#elif FILTER_ANGLE
      if (gaus_angle > CUDART_PI_F || gaus_angle < -CUDART_PI_F) {
        continue;
      }
#elif FILTER_SCALE
  #ifdef FILTER_SCALE_BOTH
      if (abs(gaus_scale) > val_offset_scale) {
        continue;
      }
  #else
      if (gaus_scale_ratio < 0) {
        continue;
      }
  #endif
#endif
      float gaus_angle_cos = gaus_cos_angles[i_gau], gaus_angle_sin = gaus_sin_angles[i_gau];
      float new_offset_x = (gaus_angle_cos * val_offset_x - gaus_angle_sin * val_offset_y) * gaus_scale_ratio;
      float new_offset_y = (gaus_angle_sin * val_offset_x + gaus_angle_cos * val_offset_y) * gaus_scale_ratio;

      int64_t w_in = w_out - lrintf(new_offset_x);
      int64_t h_in = h_out - lrintf(new_offset_y);
      bool inside = (h_in >= 0 && h_in < height_in && w_in >= 0 && w_in < width_in);

      Dtype cur_grad_gaus_weight = 0.;
      Dtype val_gaus_weight = gaus_weight[i_gau];
#ifdef USE_BACKWARD_SIDE_BALANCE
      Dtype val_grad_out = *grad_out * abs(val_offset_scale + gaus_scale) / max(val_offset_scale, 1.0);
#else
      Dtype val_grad_out = *grad_out;
#endif

      if (inside) {
        int64_t in_index = ((i_samp * num_channel + i_channel) * height_in + h_in) * width_in + w_in;
        if (UseGradIn) {
          atomicAdd(
            grad_in + in_index,
            val_grad_out * val_gaus_weight
          );
        }
        if (UseGradWeight || UseGradOffsets) {
          if (MinusCenter) {
            cur_grad_gaus_weight = val_grad_out * (data_in[in_index] - center_value);
          } else {
            cur_grad_gaus_weight = val_grad_out * data_in[in_index];
          }
        }
      } else {
        if (UseGradWeight || UseGradOffsets) {
          if (MinusCenter) {
            cur_grad_gaus_weight = val_grad_out * (fill - center_value);
          } else {
            cur_grad_gaus_weight = val_grad_out * fill;
          }
        }
      }
      // ***temp: exclude data_in
      // cur_grad_gaus_weight = *grad_out;

      if (UseGradWeight) {
        atomicAdd(
          grad_gaus_weight + i_gau,
          cur_grad_gaus_weight
        );
      }

      if (UseGradOffsets) {
        // // Use offset different
        // float offoff_x = new_offset_x - val_offset_x;
        // float offoff_y = new_offset_y - val_offset_y;
        // float anglescale_norm = hypot(gaus_angle * val_offset_scale, gaus_scale);
        // float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight * anglescale_norm / hypot(offoff_x, offoff_y);
        // grad_x += cur_grad_off_base * offoff_x;
        // grad_y += cur_grad_off_base * offoff_y;

        // Use angle and scale
        float cur_grad_off_base = cur_grad_gaus_weight * val_gaus_weight;
        grad_x += cur_grad_off_base /
          (gaus_scale * val_offset_x / val_offset_scale + gaus_angle * (-val_offset_y));
        grad_y += cur_grad_off_base /
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

#ifndef SEP_SAMPLE
    }
    if (UseGradOffsets) {
      if (channel_per_offset != 1) {
        atomicAdd(grad_offsets_x, grad_x);
        atomicAdd(grad_offsets_y, grad_y);
      } else {
        *grad_offsets_x = grad_x;
        *grad_offsets_y = grad_y;
      }
    }
#else
    } while (0);
    if (UseGradOffsets) {
      atomicAdd(grad_offsets_x, grad_x);
      atomicAdd(grad_offsets_y, grad_y);
    }
#endif
    
  }
}

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
    float fill) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
#ifndef SEP_SAMPLE
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
#else
  int64_t num_kernel = batch_size * num_channel * height_out * width_out * gaus_weight.size(1);
#endif

  DISPATCH_TWO_BOOLS(GRAD_IN_HAS_VALUE, grad_in.has_value(), GRAD_GAUS_WEIGHT_HAS_VALUE, grad_gaus_weight.has_value(), ([&] {
    DISPATCH_BOOL(GRAD_OFFSETS_HAS_VALUE, grad_offsets_x.has_value(), ([&] {
      if (GRAD_IN_HAS_VALUE || GRAD_GAUS_WEIGHT_HAS_VALUE || GRAD_OFFSETS_HAS_VALUE) {
        AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_gaus_backward_cuda", ([&] {
          displace_gaus_backward_cuda_kernel<GRAD_IN_HAS_VALUE, GRAD_GAUS_WEIGHT_HAS_VALUE, GRAD_OFFSETS_HAS_VALUE, false> <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
            num_kernel, num_channel,
            data_in.data<scalar_t>(), GRAD_IN_HAS_VALUE ? grad_in.value().data<scalar_t>() : nullptr,
            height_in, width_in,
            offsets_x.data<float>(), offsets_y.data<float>(),
            GRAD_OFFSETS_HAS_VALUE? grad_offsets_x.value().data<float>() : nullptr,
            GRAD_OFFSETS_HAS_VALUE? grad_offsets_y.value().data<float>() : nullptr,
            channel_per_offset,
            grad_out.data<scalar_t>(), height_out, width_out,
            gaus_angles.data<float>(), gaus_scales.data<float>(),
            gaus_weight.data<scalar_t>(), GRAD_GAUS_WEIGHT_HAS_VALUE ? grad_gaus_weight.value().data<scalar_t>() : nullptr,
            gaus_cos_angles.data<float>(), gaus_sin_angles.data<float>(), 
            gaus_weight.size(1), fill);
        }));
      }
    }));
  }));

  gpuErrchk(cudaGetLastError());
}

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
    float fill) {
  int64_t batch_size = grad_out.size(0);
  int64_t num_channel = grad_out.size(1);
  int64_t height_out = grad_out.size(2);
  int64_t width_out = grad_out.size(3);
  int64_t height_in = data_in.size(2);
  int64_t width_in = data_in.size(3);
#ifndef SEP_SAMPLE
  int64_t num_kernel = batch_size * num_channel * height_out * width_out;
#else
  int64_t num_kernel = batch_size * num_channel * height_out * width_out * gaus_weight.size(1);
#endif

  DISPATCH_TWO_BOOLS(GRAD_GAUS_WEIGHT_HAS_VALUE, grad_gaus_weight.has_value(), GRAD_OFFSETS_HAS_VALUE, grad_offsets_x.has_value(), ([&] {
    if (GRAD_GAUS_WEIGHT_HAS_VALUE || GRAD_OFFSETS_HAS_VALUE) {
      AT_DISPATCH_FLOATING_TYPES(data_in.type(), "displace_gaus_simple_backward_cuda", ([&] {
        displace_gaus_backward_cuda_kernel<false, GRAD_GAUS_WEIGHT_HAS_VALUE, GRAD_OFFSETS_HAS_VALUE, true> <<<GET_BLOCKS(num_kernel), CUDA_NUM_THREADS, 0, stream>>> (
          num_kernel, num_channel,
          data_in.data<scalar_t>(), (scalar_t*)nullptr,
          height_in, width_in,
          offsets_x.data<float>(), offsets_y.data<float>(),
          GRAD_OFFSETS_HAS_VALUE? grad_offsets_x.value().data<float>() : nullptr,
          GRAD_OFFSETS_HAS_VALUE? grad_offsets_y.value().data<float>() : nullptr,
          channel_per_offset,
          grad_out.data<scalar_t>(), height_out, width_out,
          gaus_angles.data<float>(), gaus_scales.data<float>(),
          gaus_weight.data<scalar_t>(), GRAD_GAUS_WEIGHT_HAS_VALUE ? grad_gaus_weight.value().data<scalar_t>() : nullptr,
          gaus_cos_angles.data<float>(), gaus_sin_angles.data<float>(), 
          gaus_weight.size(1), fill);
      }));
    }
  }));

  gpuErrchk(cudaGetLastError());
}
