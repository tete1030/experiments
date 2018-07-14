#ifndef _LACORR2D_KERNEL_CUH
#define _LACORR2D_KERNEL_CUH
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <vector>

#ifdef FLOAT_ONLY
    #if FLOAT_ONLY
        #undef FLOAT_ONLY
        #define FLOAT_ONLY 1
    #else
        #undef FLOAT_ONLY
        #define FLOAT_ONLY 0
    #endif
#else
    #define FLOAT_ONLY 0
#endif

#ifdef DEBUG 
#define D(x) x
#else 
#define D(x)
#endif

#define INDEX2D(X, Y, WIDTH) ((Y) * (WIDTH) + (X))

template <typename scalar_t>
__global__ void lacorr2d_forward_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* output,
    const int kernel_height,
    const int kernel_width,
    const int stride_height,
    const int stride_width,
    const int n_corr_h,
    const int n_corr_w,
    const int total_channel,
    const int height,
    const int width);

std::vector<at::Tensor> lacorr2d_forward_cuda(
    at::Tensor input,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width);

template <typename scalar_t>
__global__ void lacorr2d_backward_cuda_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ grad_output,
    scalar_t* grad_input,
    const int kernel_height,
    const int kernel_width,
    const int stride_height,
    const int stride_width,
    const int n_corr_h,
    const int n_corr_w,
    const int total_channel,
    const int height,
    const int width);

std::vector<at::Tensor> lacorr2d_backward_cuda(
    at::Tensor input,
    at::Tensor grad_output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width);
#endif