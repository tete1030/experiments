#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define INDEX2D(X, Y, WIDTH) ((Y) * (WIDTH) + (X))
#define FLOAT_ONLY 1

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
    const int width) {
        extern __shared__ unsigned char s[];

        int i_channel = blockIdx.x * blockDim.y + threadIdx.y;
        if (i_channel < total_channel) {
            int i_out_rel = threadIdx.x;
            int i_corr = blockIdx.y;

            int y_out_rel = i_out_rel / kernel_width;
            int x_out_rel = i_out_rel % kernel_width;

            int y_corr = i_corr / n_corr_w;
            int x_corr = i_corr % n_corr_w;

            // left and top conner of current corr in input image
            int left = stride_width * x_corr;
            int top = stride_height * y_corr;

            int half_kh = kernel_height / 2;
            int half_kw = kernel_width / 2;

            int i_inp_smem = INDEX2D(2*x_out_rel, 2*y_out_rel, 2*kernel_width);
            int y_inp = top - half_kh + 2*y_out_rel;
            int x_inp = left - half_kw + 2*x_out_rel;
            int i_inp = INDEX2D(x_inp, y_inp, width);

            input += i_channel * height * width;

            scalar_t *inp_smem = reinterpret_cast<scalar_t*>(s) + threadIdx.y * 2*kernel_height * 2*kernel_width;

            for (char y_off = 0; y_off < 2; y_off++) {
                for (char x_off = 0; x_off < 2; x_off++) {
                    if ((y_inp + y_off) < 0 || (y_inp + y_off) >= height || (x_inp + x_off) < 0 || (x_inp + x_off) >= width) {
                        inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off] = 0;
                    } else {
                        inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off] = input[i_inp + y_off * width + x_off];
                    }
                }
            }

            __syncthreads();

            scalar_t out_reg = 0.;
            for (y_inp=0; y_inp < kernel_height; y_inp++) {
                for (x_inp=0; x_inp < kernel_width; x_inp++) {
                    out_reg += inp_smem[INDEX2D(x_out_rel+x_inp, y_out_rel+y_inp, 2*kernel_width)] * inp_smem[INDEX2D(half_kw+x_inp, half_kh+y_inp, 2*kernel_width)];
                }
            }

            output += (((i_channel * n_corr_h + y_corr) * n_corr_w + x_corr) * kernel_height + y_out_rel) * kernel_width + x_out_rel;

            *output = out_reg;
        } else {
            __syncthreads();
        }
}

// cc61:
// - maximum threads 1024 per block
// - maximum resident threads 2048 per SM
// - maximum resident blocks 32 per SM
// - maximum resident warps 64 per SM
// 
// gtx1080: 20 SM

std::vector<at::Tensor> lacorr2d_forward_cuda(
    at::Tensor input,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width) {
    const int batch_size = input.size(0);
    const int channel_size = input.size(1);
    const int total_channel = batch_size * channel_size;
    const int height = input.size(2);
    const int width = input.size(3);

    AT_ASSERT(kernel_width <= width, "kernel_width must be lesser than or equal to width")
    AT_ASSERT(kernel_height <= height, "kernel_height must be lesser than or equal to height")
    AT_ASSERT(stride_width <= width, "stride_width must be lesser than or equal to width")
    AT_ASSERT(stride_height <= height, "stride_height must be lesser than or equal to height")
#if FLOAT_ONLY
    AT_ASSERT(input.type().scalarType() == at::ScalarType::Float, "input.scalarType must be float")
#endif

    const int n_corr_w = (width - kernel_width) / stride_width + 1;
    const int n_corr_h = (height - kernel_height) / stride_height + 1;
    const int n_corr = n_corr_w * n_corr_h;
    const int kernel_size = kernel_height * kernel_width;

    // work on pytorch 0.4.0 , have been changed in master 07/20/2018
    auto output = at::zeros(input.type(), std::vector<int64_t>{batch_size, channel_size, n_corr_h, n_corr_w, kernel_height, kernel_width});
    
    // ASSUME: kernel_size less than 1024, should be factor of 1024
    const int n_channel_per_block = 1024 / kernel_size;

    // std::cout << "kernel_size: " << kernel_size << std::endl;
    // std::cout << "n_channel_per_block: " << n_channel_per_block << std::endl;
    // std::cout << "blocks.x: " << ((total_channel + n_channel_per_block - 1) / n_channel_per_block) << std::endl;
    // std::cout << "blocks.y: " << n_corr << std::endl;

    // n_channel_per_block*4*kernel_size*sizeof(scalar_t) is guaranteed to be less than or equal to 32768
    // when block per SM == 2

    const dim3 threads_per_block(kernel_size, n_channel_per_block);
    const dim3 blocks((total_channel + n_channel_per_block - 1) / n_channel_per_block, n_corr);

#define CALL_FORWARD() \
    lacorr2d_forward_cuda_kernel<scalar_t><<<blocks, threads_per_block, n_channel_per_block*4*kernel_size*sizeof(scalar_t)>>>( \
        input.data<scalar_t>(), \
        output.data<scalar_t>(), \
        kernel_height, \
        kernel_width, \
        stride_height, \
        stride_width, \
        n_corr_h, \
        n_corr_w, \
        total_channel, \
        height, \
        width);

#if FLOAT_ONLY
    using scalar_t = float;
    CALL_FORWARD()
#else
    AT_DISPATCH_FLOATING_TYPES(input.type(), "lacorr2d_forward_cuda", ([&] {
        CALL_FORWARD()
    }));
#endif

    return {output};
}

// align thread id with warp/halfwarp
// align lines of images ?
// avoid bank confilict

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
    const int width) {
        extern __shared__ unsigned char s[];

        int i_channel = blockIdx.x * blockDim.y + threadIdx.y;
        
        int i_out_rel = threadIdx.x;
        int i_corr = blockIdx.y;

        int y_out_rel = i_out_rel / kernel_width;
        int x_out_rel = i_out_rel % kernel_width;

        int y_corr = i_corr / n_corr_w;
        int x_corr = i_corr % n_corr_w;

        // left and top conner of current corr in input image
        int left = stride_width * x_corr;
        int top = stride_height * y_corr;

        int half_kh = kernel_height / 2;
        int half_kw = kernel_width / 2;

        int i_inp_smem = INDEX2D(2*x_out_rel, 2*y_out_rel, 2*kernel_width);
        int y_inp = top - half_kh + 2*y_out_rel;
        int x_inp = left - half_kw + 2*x_out_rel;
        int i_inp = INDEX2D(x_inp, y_inp, width);

        input += i_channel * height * width;
        grad_input += i_channel * height * width;

        scalar_t *inp_smem = reinterpret_cast<scalar_t*>(s) + threadIdx.y * 2*kernel_height * 2*kernel_width;
        scalar_t *grad_inp_smem = inp_smem + blockDim.y * 2*kernel_height * 2*kernel_width;

        char y_off, x_off;
        if (i_channel < total_channel) {
            for (y_off = 0; y_off < 2; y_off++) {
                for (x_off = 0; x_off < 2; x_off++) {
                    if ((y_inp + y_off) < 0 || (y_inp + y_off) >= height || (x_inp + x_off) < 0 || (x_inp + x_off) >= width) {
                        inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off] = 0.;
                    } else {
                        inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off] = input[i_inp + y_off * width + x_off];
                        grad_inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off] = 0.;
                    }
                }
            }

        }

        __syncthreads();

        scalar_t grad_out_reg;

        if (i_channel < total_channel) {
            grad_out_reg = *(grad_output + (((i_channel * n_corr_h + y_corr) * n_corr_w + x_corr) * kernel_height + y_out_rel) * kernel_width + x_out_rel);

            for (int y_inp_k=0; y_inp_k < kernel_height; y_inp_k++) {
                for (int x_inp_k=0; x_inp_k < kernel_width; x_inp_k++) {
                    // atomicAdd is in case of overlapping maps
                    atomicAdd(&grad_inp_smem[INDEX2D(x_out_rel+x_inp_k, y_out_rel+y_inp_k, 2*kernel_width)], grad_out_reg * inp_smem[INDEX2D(half_kw+x_inp_k, half_kh+y_inp_k, 2*kernel_width)]);
                    // grad_inp_smem[INDEX2D(x_out_rel+x_inp_k, y_out_rel+y_inp_k, 2*kernel_width)] += grad_out_reg * inp_smem[INDEX2D(half_kw+x_inp_k, half_kh+y_inp_k, 2*kernel_width)];
                }
            }
        }

        __syncthreads();

        if (i_channel < total_channel) {
            for (y_off = 0; y_off < 2; y_off++) {
                for (x_off = 0; x_off < 2; x_off++) {
                    if ((y_inp + y_off) >= 0 && (y_inp + y_off) < height && (x_inp + x_off) >= 0 && (x_inp + x_off) < width) {
                        //grad_input[i_inp + y_off * width + x_off] = grad_inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off];
                        atomicAdd(&grad_input[i_inp + y_off * width + x_off], grad_inp_smem[i_inp_smem + y_off * 2*kernel_width + x_off]);
                    }
                }
            }
        }

        __syncthreads();

        // Use grad_inp_smem as grad_out_smem
        #define grad_out_smem grad_inp_smem

        if (i_channel < total_channel) {
            grad_out_smem[i_out_rel] = grad_out_reg;
        }

        __syncthreads();
        
        scalar_t grad_inp_reg = 0.;
        if (i_channel < total_channel) {
            for (int y_inp_k=0; y_inp_k < kernel_height; y_inp_k++) {
                for (int x_inp_k=0; x_inp_k < kernel_width; x_inp_k++) {
                    grad_inp_reg += grad_out_smem[INDEX2D(x_inp_k, y_inp_k, kernel_width)] * inp_smem[INDEX2D(x_inp_k+x_out_rel, y_inp_k+y_out_rel, 2*kernel_width)];
                }
            }
            // atomicAdd is in case of overlapping maps
            atomicAdd(&grad_input[INDEX2D(left+x_out_rel, top+y_out_rel, width)], grad_inp_reg);
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
    const int total_channel = batch_size * channel_size;
    const int height = input.size(2);
    const int width = input.size(3);

    AT_ASSERT(kernel_width <= width, "kernel_width must be lesser than or equal to width")
    AT_ASSERT(kernel_height <= height, "kernel_height must be lesser than or equal to height")
    AT_ASSERT(stride_width <= width, "stride_width must be lesser than or equal to width")
    AT_ASSERT(stride_height <= height, "stride_height must be lesser than or equal to height")
#if FLOAT_ONLY
    AT_ASSERT(input.type().scalarType() == at::ScalarType::Float, "input.scalarType must be float")
    AT_ASSERT(grad_output.type().scalarType() == at::ScalarType::Float, "grad_output.scalarType must be float")
#endif

    const int n_corr_w = (width - kernel_width) / stride_width + 1;
    const int n_corr_h = (height - kernel_height) / stride_height + 1;
    const int n_corr = n_corr_w * n_corr_h;
    const int kernel_size = kernel_height * kernel_width;

    auto grad_input = at::zeros_like(input);

    // ASSUME: kernel_size less than 1024, should be factor of 1024
    const int n_channel_per_block = 1024 / kernel_size;

    // std::cout << "kernel_size: " << kernel_size << std::endl;
    // std::cout << "n_channel_per_block: " << n_channel_per_block << std::endl;
    // std::cout << "blocks.x: " << ((total_channel + n_channel_per_block - 1) / n_channel_per_block) << std::endl;
    // std::cout << "blocks.y: " << n_corr << std::endl;

    // n_channel_per_block*4*kernel_size*sizeof(scalar_t) is guaranteed to be less than or equal to 32768
    // when block per SM == 2

    // WARN: there is not sufficient shared memory for double as scalar_t

    const dim3 threads_per_block(kernel_size, n_channel_per_block);
    const dim3 blocks((total_channel + n_channel_per_block - 1) / n_channel_per_block, n_corr);

#define CALL_BACKWARD() \
    lacorr2d_backward_cuda_kernel<scalar_t><<<blocks, threads_per_block, 2*n_channel_per_block*4*kernel_size*sizeof(scalar_t)>>>( \
        input.data<scalar_t>(), \
        grad_output.data<scalar_t>(), \
        grad_input.data<scalar_t>(), \
        kernel_height, \
        kernel_width, \
        stride_height, \
        stride_width, \
        n_corr_h, \
        n_corr_w, \
        total_channel, \
        height, \
        width);

#if FLOAT_ONLY
    using scalar_t = float;
    CALL_BACKWARD()
#else
    AT_DISPATCH_FLOATING_TYPES(input.type(), "lacorr2d_backward_cuda", ([&] {
        CALL_BACKWARD()
    }));
#endif    

    return {grad_input};
}