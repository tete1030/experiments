#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
    const int channel_size,
    const int height,
    const int width,
    const int64_t state_size) {
        const int64_t cudakernel_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (cudakernel_id < state_size) {
            const int corr_size = kernel_height * kernel_width;
            // ith sample in batch
            const int i_samp = blockIdx.z;
            // number of corr
            const int n_corr = n_corr_h * n_corr_w;
            // number of total pixels of corrs in each channel
            const int all_corr_size =  n_corr * corr_size;
            // ith channel
            const int i_channel = cudakernel_id / all_corr_size;
            // ith pixel in each channel
            const int i_all_corr = cudakernel_id % all_corr_size;
            // ith corr map in each channel
            const int i_corr = i_all_corr / corr_size;
            const int i_corr_h = i_corr / n_corr_w;
            const int i_corr_w = i_corr % n_corr_w;
            // left and top conner of current corr in input image
            const int left = stride_width * i_corr_w;
            const int top = stride_height * i_corr_h;

            // ith flatten pixel in each corr
            const int pos_kernel = i_all_corr % corr_size;
            // location in output
            const int y_out = pos_kernel / kernel_width + top;
            const int x_out = pos_kernel % kernel_width + left;
            // location in input for kernel use
            const int y_inp_k = blockIdx.y / kernel_width + top;
            const int x_inp_k = blockIdx.y % kernel_width + left;

            // location in input for multiplicand of kernel
            // (*_out - kernel_* / 2) : left/top conner of kernel projected on the input
            // (*_inp_k - top/left) : x/y inside kernel
            const int y_inp_bg = y_out - kernel_height / 2 + y_inp_k - top;
            const int x_inp_bg = x_out - kernel_width / 2 + x_inp_k - left;

            // pad 0 for multiplicand of kernel
            if (y_inp_bg < 0 || y_inp_bg >= height || x_inp_bg < 0 || x_inp_bg >= width) {
                return;
            }

            const int64_t index_out = i_samp * state_size + cudakernel_id;
            const int64_t index_bg = i_samp * channel_size * height * width + i_channel * height * width + y_inp_bg * width + x_inp_bg;
            const int64_t index_k = i_samp * channel_size * height * width + i_channel * height * width + y_inp_k * width + x_inp_k;
            auto out = output + index_out;
            auto bg_val = *(input + index_bg);
            auto k_val = *(input + index_k);
            
            atomicAdd(out, bg_val * k_val);
        }
}

std::vector<at::Tensor> lacorr2d_forward_cuda(
    at::Tensor input,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width) {
    const int batch_size = input.size(0);
    const int channel_size = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    AT_ASSERT(kernel_width <= width, "kernel_width must be lesser than or equal to width")
    AT_ASSERT(kernel_height <= height, "kernel_height must be lesser than or equal to height")
    AT_ASSERT(stride_width <= width, "stride_width must be lesser than or equal to width")
    AT_ASSERT(stride_height <= height, "stride_height must be lesser than or equal to height")

    const int n_corr_w = (width - kernel_width) / stride_width + 1;
    const int n_corr_h = (height - kernel_height) / stride_height + 1;
    const int n_corr = n_corr_w * n_corr_h;
    const int corr_size = kernel_height * kernel_width;

    // working on pytorch 0.4.0 , have been changed in master 07/20/2018
    auto output = at::zeros(input.type(), std::vector<int64_t>{batch_size, channel_size, n_corr_h, n_corr_w, kernel_height, kernel_width});
    const int64_t state_size = (int64_t)channel_size * n_corr * corr_size;

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, corr_size, batch_size);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "lacorr2d_forward_cuda", ([&] {
        lacorr2d_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            output.data<scalar_t>(),
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            n_corr_h,
            n_corr_w,
            channel_size,
            height,
            width,
            state_size);
    }));
    return {output};
}

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
    const int channel_size,
    const int height,
    const int width,
    const int64_t state_size) {
        const int64_t cudakernel_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (cudakernel_id < state_size) {
            const int corr_size = kernel_height * kernel_width;
            // ith sample in batch
            const int i_samp = blockIdx.z;
            // number of corr
            const int n_corr = n_corr_h * n_corr_w;
            // number of total pixels of corrs in each channel
            const int all_corr_size =  n_corr * corr_size;
            // ith channel
            const int i_channel = cudakernel_id / all_corr_size;
            // ith pixel in each channel
            const int i_all_corr = cudakernel_id % all_corr_size;
            // ith corr map in each channel
            const int i_corr = i_all_corr / corr_size;
            const int i_corr_h = i_corr / n_corr_w;
            const int i_corr_w = i_corr % n_corr_w;
            // left and top conner of current corr in input image
            const int left = stride_width * i_corr_w;
            const int top = stride_height * i_corr_h;

            // ith flatten pixel in each corr
            const int pos_kernel = i_all_corr % corr_size;
            // location in output
            const int y_out = pos_kernel / kernel_width + top;
            const int x_out = pos_kernel % kernel_width + left;
            // location in input for kernel use
            const int y_inp_k = blockIdx.y / kernel_width + top;
            const int x_inp_k = blockIdx.y % kernel_width + left;

            // location in input for multiplicand of kernel
            // (*_out - kernel_* / 2) : left/top conner of kernel projected on the input
            // (*_inp_k - top/left) : x/y inside kernel
            const int y_inp_bg = y_out - kernel_height / 2 + y_inp_k - top;
            const int x_inp_bg = x_out - kernel_width / 2 + x_inp_k - left;

            // pad 0 for multiplicand of kernel
            if (y_inp_bg < 0 || y_inp_bg >= height || x_inp_bg < 0 || x_inp_bg >= width) {
                return;
            }

            const int64_t index_out = i_samp * state_size + cudakernel_id;
            const int64_t index_bg = i_samp * channel_size * height * width + i_channel * height * width + y_inp_bg * width + x_inp_bg;
            const int64_t index_k = i_samp * channel_size * height * width + i_channel * height * width + y_inp_k * width + x_inp_k;
            auto grad_out_val = *(grad_output + index_out);
            auto inp_bg_val = *(input + index_bg);
            auto inp_k_val = *(input + index_k);
            auto grad_inp_bg = grad_input + index_bg;
            auto grad_inp_k = grad_input + index_k;
            
            atomicAdd(grad_inp_bg, grad_out_val * inp_k_val);
            atomicAdd(grad_inp_k, grad_out_val * inp_bg_val);
        }
}

std::vector<at::Tensor> lacorr2d_backward_cuda(
    at::Tensor input,
    at::Tensor grad_output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width) {
    const int batch_size = input.size(0);
    const int channel_size = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    AT_ASSERT(kernel_width <= width, "kernel_width must be lesser than or equal to width")
    AT_ASSERT(kernel_height <= height, "kernel_height must be lesser than or equal to height")
    AT_ASSERT(stride_width <= width, "stride_width must be lesser than or equal to width")
    AT_ASSERT(stride_height <= height, "stride_height must be lesser than or equal to height")

    const int n_corr_w = (width - kernel_width) / stride_width + 1;
    const int n_corr_h = (height - kernel_height) / stride_height + 1;
    const int n_corr = n_corr_w * n_corr_h;
    const int corr_size = kernel_height * kernel_width;

    const int64_t state_size = (int64_t)channel_size * n_corr * corr_size;

    auto grad_input = at::zeros_like(input);

    const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, corr_size, batch_size);
    AT_DISPATCH_FLOATING_TYPES(input.type(), "lacorr2d_forward_cuda", ([&] {
        lacorr2d_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data<scalar_t>(),
            grad_output.data<scalar_t>(),
            grad_input.data<scalar_t>(),
            kernel_height,
            kernel_width,
            stride_height,
            stride_width,
            n_corr_h,
            n_corr_w,
            channel_size,
            height,
            width,
            state_size);
    }));
    return {grad_input};
}