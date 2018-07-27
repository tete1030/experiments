#include <torch/torch.h>
#include <vector>
#include "lacorr2d_kernel.cuh"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lacorr2d_forward(
    at::Tensor input,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_top,
    int padding_bottom,
    int padding_left,
    int padding_right) {
        CHECK_INPUT(input);

        return lacorr2d_forward_cuda(input,
                                     kernel_height,
                                     kernel_width,
                                     stride_height,
                                     stride_width,
                                     padding_top,
                                     padding_bottom,
                                     padding_left,
                                     padding_right);
}

std::vector<at::Tensor> lacorr2d_backward(
    at::Tensor input,
    at::Tensor grad_output,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_top,
    int padding_bottom,
    int padding_left,
    int padding_right) {
        CHECK_INPUT(input);
        CHECK_INPUT(grad_output);

        return lacorr2d_backward_cuda(input,
                                      grad_output,
                                      kernel_height,
                                      kernel_width,
                                      stride_height,
                                      stride_width,
                                      padding_top,
                                      padding_bottom,
                                      padding_left,
                                      padding_right);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lacorr2d_forward", &lacorr2d_forward, "lacorr2d forward");
  m.def("lacorr2d_backward", &lacorr2d_backward, "lacorr2d backward");
}