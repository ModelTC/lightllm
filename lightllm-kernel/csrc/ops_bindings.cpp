#include <torch/extension.h>
#include "ops_common.h"
#include <pybind11/pybind11.h>

namespace lightllm {
namespace ops {

PYBIND11_MODULE(_C, m) {
    m.def("grouped_topk", &grouped_topk,
          "Grouped Top-K routing (CUDA)",
          py::arg("topk_weights"),
          py::arg("correction_bias"),
          py::arg("topk_indices"),
          py::arg("group_indices"),
          py::arg("gating_output"),
          py::arg("num_expert_group"),
          py::arg("topk_group"),
          py::arg("topk"),
          py::arg("renormalize"),
          py::arg("scoring_func"),
          py::arg("group_scores") = torch::Tensor());
    m.def("rmsnorm_align16_bf16", &rmsnorm_align16_bf16, "RMSNORM (CUDA)");
    m.def("pre_tp_norm_bf16", &pre_tp_norm_bf16, "PRE TP NORM (CUDA)");
    m.def("post_tp_norm_bf16", &post_tp_norm_bf16, "POST TP NORM (CUDA)");
    m.def("per_token_quant_bf16_fp8", &per_token_quant_bf16_fp8, "PER TOKEN QUANT (CUDA)");
    m.def("add_norm_quant_bf16_fp8", &add_norm_quant_bf16_fp8, "ADD NORM QUANT FUSED (CUDA)");
    m.def("gelu_per_token_quant_bf16_fp8", &gelu_per_token_quant_bf16_fp8, "GELU QUANT FUSED (CUDA)");
    m.def("cutlass_scaled_mm", &cutlass_scaled_mm, "CUTLASS SCALED MM (CUDA)");
}

} // namespace ops
} // namespace lightllm