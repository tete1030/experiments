#include "displace.h"

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

void displace_frac_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_frac_forward_cuda(stream, data_in, offsets, channel_per_offset, data_out);
}

void displace_frac_backward(
    const int64_t state,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_frac_backward_cuda(stream, grad_in, offsets, channel_per_offset, grad_out);
}

void displace_frac_offset_backward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  CHECK_INPUT(grad_offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  AT_ASSERTM(grad_offsets.dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);

  displace_frac_offset_backward_cuda(stream, data_in, offsets, grad_offsets, channel_per_offset, grad_out);
}

void displace_pos_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor data_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_forward_cuda(stream, data_in, offsets, channel_per_offset, data_out);
}

void displace_pos_sep_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    at::Tensor data_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  AT_ASSERTM( \
    (offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float) || \
    (offsets_x.dtype() == at::ScalarType::Int && offsets_y.dtype() == at::ScalarType::Int), \
    "dtype of offsets must be float or int");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_forward_cuda(stream, data_in, offsets_x, offsets_y, channel_per_offset, data_out);
}

void displace_pos_backward(
    const int64_t state,
    const at::Tensor data_in, at::Tensor grad_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  CHECK_INPUT(grad_offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  AT_ASSERTM(grad_offsets.dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_backward_cuda(stream, data_in, grad_in, offsets, grad_offsets, channel_per_offset, grad_out);
}

void displace_pos_sep_backward(
    const int64_t state,
    const at::optional<at::Tensor> data_in, at::Tensor grad_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    at::optional<at::Tensor> grad_offsets_x,
    at::optional<at::Tensor> grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  if (data_in) {
    CHECK_INPUT(data_in.value());
  }
  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  if (grad_offsets_x) {
    CHECK_INPUT(grad_offsets_x.value());
    CHECK_INPUT(grad_offsets_y.value());
  }
  AT_ASSERTM( \
    (offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float) || \
    (offsets_x.dtype() == at::ScalarType::Int && offsets_y.dtype() == at::ScalarType::Int), \
    "dtype of offsets must be float or int");
  if (offsets_x.dtype() == at::ScalarType::Float) {
    AT_ASSERTM(data_in && grad_offsets_x && grad_offsets_y, "data_in, grad_offsets_x, grad_offsets_y must have value")
    AT_ASSERTM(grad_offsets_x.value().dtype() == at::ScalarType::Float && grad_offsets_y.value().dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  }
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_backward_cuda(stream, data_in, grad_in, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, channel_per_offset, grad_out);
}

void displace_pos_backward_data(
    const int64_t state,
    at::Tensor grad_in,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);

  displace_pos_backward_data_cuda(stream, grad_in, offsets, channel_per_offset, grad_out);
}

void displace_pos_sep_backward_data(
    const int64_t state,
    at::Tensor grad_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  AT_ASSERTM(offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);

  displace_pos_backward_data_cuda(stream, grad_in, offsets_x, offsets_y, channel_per_offset, grad_out);
}

void displace_pos_backward_offset(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets,
    at::Tensor grad_offsets,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets);
  CHECK_INPUT(grad_offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  AT_ASSERTM(grad_offsets.dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_backward_offset_cuda(stream, data_in, offsets, grad_offsets, channel_per_offset, grad_out);
}

void displace_pos_sep_backward_offset(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    at::Tensor grad_offsets_x,
    at::Tensor grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  CHECK_INPUT(grad_offsets_x);
  CHECK_INPUT(grad_offsets_y);
  AT_ASSERTM(offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  AT_ASSERTM(grad_offsets_x.dtype() == at::ScalarType::Float && grad_offsets_y.dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_pos_backward_offset_cuda(stream, data_in, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, channel_per_offset, grad_out);
}

void offset_mask_frac(
    const int64_t state,
    const at::Tensor input,
    const at::Tensor offsets,
    const int64_t channel_per_offset,
    at::Tensor output) {

  CHECK_INPUT(input);
  CHECK_INPUT(output);
  CHECK_INPUT(offsets);
  AT_ASSERTM(offsets.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);

  offset_mask_frac_cuda(stream, input, offsets, channel_per_offset, output);
}

void displace_gaus_forward(
    const int64_t state,
    const at::Tensor data_in,
    const at::Tensor offsets_x,
    const at::Tensor offsets_y,
    const int64_t channel_per_offset,
    at::Tensor data_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales, const at::Tensor gaus_weight,
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    // dtype
    float fill) {

  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  CHECK_INPUT(gaus_angles);
  CHECK_INPUT(gaus_scales);
  CHECK_INPUT(gaus_weight);
  CHECK_INPUT(gaus_cos_angles);
  CHECK_INPUT(gaus_sin_angles);
  AT_ASSERTM(offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  displace_gaus_forward_cuda(stream, data_in, offsets_x, offsets_y, channel_per_offset, data_out, gaus_angles, gaus_scales, gaus_weight, gaus_cos_angles, gaus_sin_angles, fill);
}

void displace_gaus_backward(
    const int64_t state,
    const at::Tensor data_in, at::optional<at::Tensor> grad_in,
    const at::Tensor offsets_x, const at::Tensor offsets_y,
    at::optional<at::Tensor> grad_offsets_x, at::optional<at::Tensor> grad_offsets_y,
    const int64_t channel_per_offset,
    const at::Tensor grad_out,
    const at::Tensor gaus_angles, const at::Tensor gaus_scales,
    const at::Tensor gaus_weight, at::optional<at::Tensor> grad_gaus_weight,
    const at::Tensor gaus_cos_angles, const at::Tensor gaus_sin_angles,
    const at::Tensor gaus_angle_stds, const at::Tensor gaus_scale_stds,
    at::optional<at::Tensor> grad_gaus_angles, at::optional<at::Tensor> grad_gaus_scales,
    // dtype
    float fill, bool simple=false) {

  CHECK_INPUT(data_in);
  if (grad_in) {
    CHECK_INPUT(grad_in.value());
  }
  CHECK_INPUT(grad_out);
  CHECK_INPUT(offsets_x);
  CHECK_INPUT(offsets_y);
  if (grad_offsets_x) {
    CHECK_INPUT(grad_offsets_x.value());
    CHECK_INPUT(grad_offsets_y.value());
  } else {
    AT_ASSERTM(!grad_offsets_y.has_value(), "grad_offsets_x and grad_offsets_y's existence should be same");
  }
  CHECK_INPUT(gaus_angles);
  CHECK_INPUT(gaus_scales);
  CHECK_INPUT(gaus_weight);
  if (grad_gaus_weight) {
    CHECK_INPUT(grad_gaus_weight.value());
  }
  CHECK_INPUT(gaus_cos_angles);
  CHECK_INPUT(gaus_sin_angles);
  if (grad_gaus_angles) {
    CHECK_INPUT(grad_gaus_angles.value());
    CHECK_INPUT(grad_gaus_scales.value());
  } else {
    AT_ASSERTM(!grad_gaus_scales.has_value(), "grad_gaus_angles and grad_gaus_scales's existence should be same");
  }
  AT_ASSERTM(offsets_x.dtype() == at::ScalarType::Float && offsets_y.dtype() == at::ScalarType::Float, "dtype of offsets must be float");
  if (grad_offsets_x) {
    AT_ASSERTM(grad_offsets_x.value().dtype() == at::ScalarType::Float && grad_offsets_y.value().dtype() == at::ScalarType::Float, "dtype of grad_offsets must be float");
  }
  if (grad_gaus_angles) {
    AT_ASSERTM(grad_gaus_angles.value().dtype() == at::ScalarType::Float && grad_gaus_scales.value().dtype() == at::ScalarType::Float, "dtype of grad_gaus_angles|scales must be float");
  }
  AT_ASSERTM(gaus_angle_stds.dtype() == at::ScalarType::Float && gaus_scale_stds.dtype() == at::ScalarType::Float, "dtype of gaus_angle|scale_stds must be float");
  AT_ASSERTM(gaus_angle_stds.size(0) == offsets_x.size(1) && gaus_scale_stds.size(0) == offsets_x.size(1), "size of gaus_angle|scale_stds must equal to num_offset");
  auto stream = THCState_getCurrentStream((THCState*)state);
  
  if (!simple) {
    displace_gaus_backward_cuda(stream, data_in, grad_in, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, channel_per_offset, grad_out, gaus_angles, gaus_scales, gaus_weight, grad_gaus_weight, gaus_cos_angles, gaus_sin_angles, gaus_angle_stds, gaus_scale_stds, grad_gaus_angles, grad_gaus_scales, fill);
  } else {
    displace_gaus_simple_backward_cuda(stream, data_in, offsets_x, offsets_y, grad_offsets_x, grad_offsets_y, channel_per_offset, grad_out, gaus_angles, gaus_scales, gaus_weight, grad_gaus_weight, gaus_cos_angles, gaus_sin_angles, gaus_angle_stds, gaus_scale_stds, grad_gaus_angles, grad_gaus_scales, fill);
  }
}

namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("displace_forward", &displace_forward, "displace forward");
  m.def("displace_backward", &displace_backward, "displace backward");
  m.def("offset_mask", &offset_mask, "offset mask");
  m.def("displace_frac_forward", &displace_frac_forward, "displace fractional forward");
  m.def("displace_frac_backward", &displace_frac_backward, "displace fractional input backward");
  m.def("displace_frac_offset_backward", &displace_frac_offset_backward, "displace fractional offset backward");
  m.def("offset_mask_frac", &offset_mask_frac, "offset mask");
  m.def("displace_pos_forward", &displace_pos_forward, "positional displace forward");
  m.def("displace_pos_backward", &displace_pos_backward, "positional displace backward");
  m.def("displace_pos_backward_data", &displace_pos_backward_data, "positional displace backward for data");
  m.def("displace_pos_backward_offset", &displace_pos_backward_offset, "positional displace backward for offset");
  m.def("displace_pos_sep_forward", &displace_pos_sep_forward, "positional displace forward");
  m.def("displace_pos_sep_backward", &displace_pos_sep_backward, "positional displace backward");
  m.def("displace_pos_sep_backward_data", &displace_pos_sep_backward_data, "positional displace backward for data");
  m.def("displace_pos_sep_backward_offset", &displace_pos_sep_backward_offset, "positional displace backward for offset");
  m.def("displace_gaus_forward", &displace_gaus_forward, "positional displace with gaussian backward for offset");
  m.def("displace_gaus_backward", &displace_gaus_backward, "positional displace with gaussian backward for offset",
    py::arg("state"),
    py::arg("data_in"), py::arg("grad_in"),
    py::arg("offsets_x"), py::arg("offsets_y"),
    py::arg("grad_offsets_x"), py::arg("grad_offsets_y"),
    py::arg("channel_per_offset"),
    py::arg("grad_out"),
    py::arg("gaus_angles"), py::arg("gaus_scales"),
    py::arg("gaus_weight"), py::arg("grad_gaus_weight"),
    py::arg("gaus_cos_angles"), py::arg("gaus_sin_angles"),
    py::arg("gaus_angle_stds"), py::arg("gaus_scale_stds"),
    py::arg("grad_gaus_angles"), py::arg("grad_gaus_scales"),
    py::arg("fill"),
    py::arg("simple") = false);
  m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input, "cudnn convolution backward for input");
  m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight, "cudnn convolution backward for weight");
  m.def("cudnn_convolution_backward_bias", &at::cudnn_convolution_backward_bias, "cudnn convolution backward for bias");
}
