#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#define INDEX2D(X, Y, WIDTH) ((Y) * (WIDTH) + (X))
#define FLOAT_ONLY 1
// CC61
#define NUM_BANK 32
#define HALF_WARP 16

#define CONDITION_INSIDE(X, Y, W, H) ((Y) >= 0 && (Y) < (H) && (X) >= 0 && (X) < (W))
#define CONDITION_UP_true(OFFSET, UBOUND) ((OFFSET) < (UBOUND)) &&
#define CONDITION_UP_false(OFFSET, UBOUND)

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

        int ichan = blockIdx.x * blockDim.y + threadIdx.y;
        
        int ithread = threadIdx.x;
        int icorr = blockIdx.y;

        int y_k = ithread / kernel_width;
        int x_k = ithread % kernel_width;

        int y_corr = icorr / n_corr_w;
        int x_corr = icorr % n_corr_w;

        // left and top conner of current corr in input image
        int left_k = stride_width * x_corr;
        int top_k = stride_height * y_corr;

        int half_kh = kernel_height / 2;
        int half_kw = kernel_width / 2;

        int bg_width = kernel_width * 2 - kernel_width % 2;
        int bg_height = kernel_height * 2 - kernel_height % 2;

        input += ichan * height * width;

        scalar_t *inp_smem = reinterpret_cast<scalar_t*>(s) + threadIdx.y * bg_width * bg_height;

        if (ichan < total_channel) {
            int i_inp_smem = INDEX2D(2*x_k, 2*y_k, bg_width);
            int y_inp = top_k - half_kh + 2*y_k;
            int x_inp = left_k - half_kw + 2*x_k;
            int i_inp = INDEX2D(x_inp, y_inp, width);

            #define INIT_INP(X_INP, Y_INP, COND_X, COND_Y) \
                if(CONDITION_UP_##COND_X(X_INP, left_k-half_kw+bg_width) CONDITION_UP_##COND_Y(Y_INP, top_k-half_kh+bg_height) true) { \
                    if (CONDITION_INSIDE((X_INP), (Y_INP), width, height)) { \
                        inp_smem[i_inp_smem] = input[i_inp]; \
                    } else { \
                        inp_smem[i_inp_smem] = 0.; \
                    } \
                }

            INIT_INP(x_inp, y_inp, false, false)
            i_inp += 1;
            i_inp_smem += 1;
            INIT_INP(x_inp+1, y_inp, true, false)
            i_inp += width - 1;
            i_inp_smem += bg_width - 1;
            INIT_INP(x_inp, y_inp+1, false, true)
            i_inp += 1;
            i_inp_smem += 1;
            INIT_INP(x_inp+1, y_inp+1, true, true)
            i_inp -= width + 1;
            i_inp_smem -= bg_width + 1;

            #undef INIT_INP

        }

        __syncthreads();

        if (ichan < total_channel) {
            scalar_t out_reg = 0.;

            int i_bg = INDEX2D(x_k, y_k, bg_width);
            int i_k = INDEX2D(half_kw, half_kh, bg_width);
            for (int y_off=0; y_off < kernel_height; y_off++) {
                for (int x_off=0; x_off < kernel_width; x_off++) {
                    out_reg += inp_smem[i_bg] * inp_smem[i_k];
                    i_bg += 1;
                    i_k += 1;
                }
                i_bg += bg_width - kernel_width;
                i_k += bg_width - kernel_width;
            }

            output += (((ichan * n_corr_h + y_corr) * n_corr_w + x_corr) * kernel_height + y_k) * kernel_width + x_k;

            *output = out_reg;
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
    const int bg_width = kernel_width * 2 - kernel_width % 2;
    const int bg_height = kernel_height * 2 - kernel_height % 2;

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
    const int shared_memory_size = n_channel_per_block * bg_width * bg_height;

#define CALL_FORWARD() \
    lacorr2d_forward_cuda_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size*sizeof(scalar_t)>>>( \
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

        int ichan = blockIdx.x * blockDim.y + threadIdx.y;
        
        int ithread = threadIdx.x;
        int icorr = blockIdx.y;

        int y_k = ithread / kernel_width;
        int x_k = ithread % kernel_width;

        int y_corr = icorr / n_corr_w;
        int x_corr = icorr % n_corr_w;

        // left and top conner of current corr in input image
        int left_k = stride_width * x_corr;
        int top_k = stride_height * y_corr;

        int half_kh = kernel_height / 2;
        int half_kw = kernel_width / 2;

        int bg_width = kernel_width * 2 - kernel_width % 2;
        int bg_height = kernel_height * 2 - kernel_height % 2;
        int bg_size = bg_width * bg_height;

        int i_inp_smem = INDEX2D(2*x_k, 2*y_k, bg_width);
        int y_inp = top_k - half_kh + 2*y_k;
        int x_inp = left_k - half_kw + 2*x_k;
        int i_inp = INDEX2D(x_inp, y_inp, width);

        input += ichan * height * width;
        grad_input += ichan * height * width;

        scalar_t *inp_smem = reinterpret_cast<scalar_t*>(s) + threadIdx.y * bg_size;
        scalar_t *grad_inp_smem = inp_smem + blockDim.y * bg_size;

        if (ichan < total_channel) {

            #define INIT_INP_GINP(X_INP, Y_INP, COND_X, COND_Y) \
                if(CONDITION_UP_##COND_X(X_INP, left_k-half_kw+bg_width) CONDITION_UP_##COND_Y(Y_INP, top_k-half_kh+bg_height) true) { \
                    if (CONDITION_INSIDE((X_INP), (Y_INP), width, height)) { \
                        inp_smem[i_inp_smem] = input[i_inp]; \
                        grad_inp_smem[i_inp_smem] = 0.; \
                    } else { \
                        inp_smem[i_inp_smem] = 0.; \
                    } \
                }

            INIT_INP_GINP(x_inp, y_inp, false, false)
            i_inp += 1;
            i_inp_smem += 1;
            INIT_INP_GINP(x_inp+1, y_inp, true, false)
            i_inp += width - 1;
            i_inp_smem += bg_width - 1;
            INIT_INP_GINP(x_inp, y_inp+1, false, true)
            i_inp += 1;
            i_inp_smem += 1;
            INIT_INP_GINP(x_inp+1, y_inp+1, true, true)
            i_inp -= width + 1;
            i_inp_smem -= bg_width + 1;

            #undef INIT_INP_GINP

        }

        __syncthreads();

        // # Updating background grad

        scalar_t grad_out_reg;

        if (ichan < total_channel) {
            grad_out_reg = *(grad_output + (((ichan * n_corr_h + y_corr) * n_corr_w + x_corr) * kernel_height + y_k) * kernel_width + x_k);

            grad_inp_smem += y_k * bg_width + x_k;
            inp_smem += half_kh * bg_width + half_kw;
            // *_off respect to input/grad_input location, output location is fixed
            for (int y_off=0; y_off < kernel_height; y_off++) {
                for (int x_off=0; x_off < kernel_width; x_off++) {
                    // atomicAdd is in case of overlapping maps
                    atomicAdd(grad_inp_smem, grad_out_reg * (*inp_smem));
                    grad_inp_smem += 1;
                    inp_smem += 1;
                }
                grad_inp_smem += bg_width - kernel_width;
                inp_smem += bg_width - kernel_width;
            }
            grad_inp_smem -= (kernel_height + y_k) * bg_width + x_k;
            inp_smem -= (kernel_height + half_kh) * bg_width + half_kw;
        }

        __syncthreads();

        if (ichan < total_channel) {

            #define STORE_GRADINP(X_INP, Y_INP, COND_X, COND_Y) \
                if (CONDITION_UP_##COND_X(X_INP, left_k-half_kw+bg_width) CONDITION_UP_##COND_Y(Y_INP, top_k-half_kh+bg_height) CONDITION_INSIDE((X_INP), (Y_INP), width, height)) { \
                    atomicAdd(&grad_input[i_inp], grad_inp_smem[i_inp_smem]); \
                }
            
            STORE_GRADINP(x_inp, y_inp, false, false)
            i_inp += 1;
            i_inp_smem += 1;
            STORE_GRADINP(x_inp+1, y_inp, true, false)
            i_inp += width - 1;
            i_inp_smem += bg_width - 1;
            STORE_GRADINP(x_inp, y_inp+1, false, true)
            i_inp += 1;
            i_inp_smem += 1;
            STORE_GRADINP(x_inp+1, y_inp+1, true, true)
            i_inp -= width + 1;
            i_inp_smem -= bg_width + 1;

            #undef STORE_GRADINP
        }

        __syncthreads();

        // Use grad_inp_smem as grad_out_smem
        #define grad_out_smem grad_inp_smem

        if (ichan < total_channel) {
            grad_out_smem[ithread] = grad_out_reg;
        }

        __syncthreads();
        
        scalar_t grad_inp_reg = 0.;
        if (ichan < total_channel) {
            inp_smem += y_k * bg_width + x_k;
            // *_off respect to output_location/input_patch_location, input offset inside each patch is fixed (input)
            for (int y_off=0; y_off < kernel_height; y_off++) {
                for (int x_off=0; x_off < kernel_width; x_off++) {
                    grad_inp_reg += (*grad_out_smem) * (*inp_smem);
                    inp_smem += 1;
                    grad_out_smem += 1;
                }
                inp_smem += bg_width - kernel_width;
                // grad_out_smem += kernel_width - kernel_width;
            }
            inp_smem -= (kernel_height + y_k) * bg_width + x_k;
            grad_out_smem -= kernel_height * kernel_width;

            // atomicAdd is used because of potential overlapping maps
            atomicAdd(&grad_input[INDEX2D(left_k+x_k, top_k+y_k, width)], grad_inp_reg);
        }

        #undef grad_out_smem
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
    const int bg_width = kernel_width * 2 - kernel_width % 2;
    const int bg_height = kernel_height * 2 - kernel_height % 2;

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
    const int shared_memory_size = 2 * n_channel_per_block * bg_width * bg_height;

#define CALL_BACKWARD() \
    lacorr2d_backward_cuda_kernel<scalar_t><<<blocks, threads_per_block, shared_memory_size*sizeof(scalar_t)>>>( \
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