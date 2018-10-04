#include <torch/torch.h>
#include <THC/THC.h>
#include <THC/THCGeneral.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_tuples.h>
#include <torch/csrc/utils/python_numbers.h>


#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void displace_forward_cuda(
    cudaStream_t stream,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out);

void displace_backward_cuda(
    cudaStream_t stream,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out);

void offset_mask_cuda(
    cudaStream_t stream,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output,
    const at::IntList side_thickness);

void displace_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Int, "dtype of offsets must be int");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_forward_cuda(stream, data_in, offsets, channel_per_offset, data_out);
}

void displace_backward(
    const int64_t state,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Int, "dtype of offsets must be int");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_backward_cuda(stream, grad_in, offsets, channel_per_offset, grad_out);
}

void offset_mask(
    const int64_t state,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output,
    const at::IntList side_thickness) {

  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Int, "dtype of offsets must be int");
  auto stream = THCState_getCurrentStream((THCState*)state);

  offset_mask_cuda(stream, input, offsets, channel_per_offset, output, side_thickness);
}

namespace pybind11 { namespace detail {
  template<> struct type_caster<at::IntList> {
  public:
    PYBIND11_TYPE_CASTER(at::IntList, _("at::IntList"));

    bool load(handle src, bool) {
      PyObject *source = src.ptr();
      auto tuple = PyTuple_Check(source);
      if (tuple || PyList_Check(source)) {
        auto size = tuple ? PyTuple_GET_SIZE(source) : PyList_GET_SIZE(source);
        v_value.resize(size);
        for (int idx = 0; idx < size; idx++) {
          PyObject* obj = tuple ? PyTuple_GET_ITEM(source, idx) : PyList_GET_ITEM(source, idx);
          if (THPVariable_Check(obj)) {
            v_value[idx] = THPVariable_Unpack(obj).toCLong();
          } else if (PyLong_Check(obj)) {
            // use THPUtils_unpackLong after it is safe to include python_numbers.h
            v_value[idx] = THPUtils_unpackLong(obj);
          } else {
            return false;
          }
        }
        value = v_value;
        return true;
      }
      return false;
    }
    static handle cast(at::IntList src, return_value_policy /* policy */, handle /* parent */) {
      return handle(THPUtils_packInt64Array(src.size(), src.data()));
    }
  private:
    std::vector<int64_t> v_value;
  };
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("displace_forward", &displace_forward, "displace forward");
  m.def("displace_backward", &displace_backward, "displace backward");
  m.def("offset_mask", &offset_mask, "offset mask");
  m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input, "cudnn convolution backward for input");
  m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight, "cudnn convolution backward for weight");
  m.def("cudnn_convolution_backward_bias", &at::cudnn_convolution_backward_bias, "cudnn convolution backward for bias");
}
