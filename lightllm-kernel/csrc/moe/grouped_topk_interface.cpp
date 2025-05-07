#include <torch/extension.h>
#include "../../include/ops_common.h"


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
    torch::Tensor group_scores = torch::Tensor());

torch::Tensor grouped_topk(
    torch::Tensor topk_weights,
    torch::Tensor correction_bias,
    torch::Tensor topk_indices,
    torch::Tensor group_indices,
    torch::Tensor gating_output,
    int  num_expert_group,
    int  topk_group,
    int  topk,
    bool renormalize,
    std::string scoring_func,
    torch::Tensor group_scores /* = {} */) {

    TORCH_CHECK(topk_weights.is_cuda(),   "topk_weights must be CUDA tensor");
    TORCH_CHECK(gating_output.is_cuda(),  "gating_output must be CUDA tensor");

    grouped_topk(topk_weights,
                 correction_bias,
                 topk_indices,
                 group_indices,
                 gating_output,
                 num_expert_group,
                 topk_group,
                 topk,
                 renormalize,
                 scoring_func,
                 group_scores);

    // 就地写结果，所以这里直接返回topk_weights
    return topk_weights;
}
