#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <tuple>

#include "utils.h"


namespace lightllm {
namespace ops {

using namespace lightllm;

int64_t meta_size();
Tensor pre_tp_norm_bf16(Tensor &input);

Tensor post_tp_norm_bf16(
    Tensor &input, const Tensor& weight,
    const Tensor& tp_variance, const int embed_dim,
    const fp32_t eps
);

Tensor rmsnorm_align16_bf16(
    const Tensor &X, const Tensor &W,
    const fp32_t eps
);

void per_token_quant_bf16_fp8(
    Tensor& output,
    const Tensor& input,
    Tensor& scales
);

void per_token_quant_bf16_int8(
    Tensor& output,
    const Tensor& input,
    Tensor& scales
);

std::tuple<Tensor, Tensor> add_norm_quant_bf16_fp8(
    Tensor& X, const Tensor &R, const Tensor &W,
    const fp32_t eps
);

void gelu_per_token_quant_bf16_fp8(
    Tensor& output,
    const Tensor& input,
    Tensor& scales
);

void cutlass_scaled_mm(
    Tensor& c, Tensor const& a,
    Tensor const& b, Tensor const& a_scales,
    Tensor const& b_scales,
    c10::optional<Tensor> const& bias,
    c10::optional<Tensor> const& ls
);

Tensor grouped_topk(
        Tensor topk_weights,
        Tensor correction_bias,
        Tensor topk_indices,
        Tensor group_indices,
        Tensor gating_output,
        int64_t  num_expert_group,
        int64_t  topk_group,
        int64_t  topk,
        bool     renormalize,
        std::string scoring_func,
        Tensor group_scores
);

void all_gather(
    int64_t _fa,
    Tensor& inp,
    Tensor& out,
    int64_t _reg_buffer,
    int64_t reg_buffer_sz_bytes
);

void group_int8kv_flashdecoding_attention(
    const int64_t seq_block_size, 
    Tensor mid_o_emb, 
    Tensor mid_o_logexpsum, 
    fp32_t att_scale, 
    Tensor q, 
    Tensor k, 
    Tensor k_s,  
    Tensor v,  
    Tensor v_s, 
    Tensor req_to_tokens, 
    Tensor b_req_idx, 
    Tensor b_seq_len, 
    int64_t max_len_in_batch);

void group_int8kv_decode_attention(
    Tensor o, 
    Tensor q, 
    Tensor k, 
    Tensor k_s,  
    Tensor v,  
    Tensor v_s, 
    Tensor req_to_tokens, 
    Tensor b_req_idx, 
    Tensor b_seq_len, 
    int64_t max_len_in_batch);

int64_t init_custom_gather_ar(
    const std::vector<int64_t>& fake_ipc_ptrs,
    torch::Tensor& rank_data,
    int64_t rank,
    bool full_nvlink
);

void allgather_dispose(
    int64_t _fa
);

void allgather_register_buffer(
    int64_t _fa,
    const std::vector<int64_t>& fake_ipc_ptrs
);

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
allgather_get_graph_buffer_ipc_meta(
    int64_t _fa
);

void allgather_register_graph_buffers(
    int64_t _fa,
    const std::vector<std::vector<int64_t>>& handles,
    const std::vector<std::vector<int64_t>>& offsets
);

} // namespace ops
} // namespace lightllm