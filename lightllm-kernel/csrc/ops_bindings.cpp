#include <torch/extension.h>
#include "../include/ops_common.h"
#include <pybind11/pybind11.h>

void grouped_topk_cuda(
        torch::Tensor& topk_weights,
        torch::Tensor& correction_bias,
        torch::Tensor& topk_indices,
        torch::Tensor& group_indices,
        torch::Tensor& gating_output,
        int  num_expert_group,
        int  topk_group,
        int  topk,
        bool renormalize,
        std::string scoring_func,
        torch::Tensor group_scores);


torch::Tensor grouped_topk(
        torch::Tensor topk_weights,
        torch::Tensor correction_bias,
        torch::Tensor topk_indices,
        torch::Tensor group_indices,
        torch::Tensor gating_output,
        int64_t  num_expert_group,
        int64_t  topk_group,
        int64_t  topk,
        bool     renormalize,
        std::string scoring_func,
        torch::Tensor group_scores) {

    grouped_topk_cuda(topk_weights, correction_bias, topk_indices, group_indices,
                      gating_output,
                      static_cast<int>(num_expert_group),
                      static_cast<int>(topk_group),
                      static_cast<int>(topk),
                      renormalize, scoring_func, group_scores);

    return topk_weights;
}

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
}