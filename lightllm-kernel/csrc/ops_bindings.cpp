#include <torch/extension.h>
#include "ops_common.h"
#include <pybind11/pybind11.h>

namespace lightllm {
namespace ops {

PYBIND11_MODULE(_C, m) {
    m.def("grouped_topk", &grouped_topk,"GROUPED TOP-K (CUDA)");
    m.def("rmsnorm_align16_bf16", &rmsnorm_align16_bf16, "RMSNORM (CUDA)");
    m.def("pre_tp_norm_bf16", &pre_tp_norm_bf16, "PRE TP NORM (CUDA)");
    m.def("post_tp_norm_bf16", &post_tp_norm_bf16, "POST TP NORM (CUDA)");
    m.def("per_token_quant_bf16_fp8", &per_token_quant_bf16_fp8, "PER TOKEN QUANT FP8 (CUDA)");
    m.def("per_token_quant_bf16_int8", &per_token_quant_bf16_int8, "PER TOKEN QUANT INT8 (CUDA)");
    m.def("add_norm_quant_bf16_fp8", &add_norm_quant_bf16_fp8, "ADD NORM QUANT FUSED (CUDA)");
    m.def("gelu_per_token_quant_bf16_fp8", &gelu_per_token_quant_bf16_fp8, "GELU QUANT FUSED (CUDA)");
    m.def("cutlass_scaled_mm", &cutlass_scaled_mm, "CUTLASS SCALED MM (CUDA)");
    m.def("all_gather", &all_gather, "ALL GATHER (CUDA)");
    m.def("allgather_dispose", &allgather_dispose, "ALL GATHER DISPOSE (CUDA)");
    m.def("init_custom_gather_ar", &init_custom_gather_ar, "INIT CUSTOM GATHER AR (CUDA)");
    m.def("allgather_register_buffer", &allgather_register_buffer, "ALL GATHER REGISTER BUFFER (CUDA)");
    m.def("allgather_register_graph_buffers", &allgather_register_graph_buffers, "ALL GATHER REGISTER BRAPH BUFFERS (CUDA)");
    m.def("allgather_get_graph_buffer_ipc_meta", &allgather_get_graph_buffer_ipc_meta, "ALL GATHER GET GRAPH BUFFER IPC META (CUDA)");
    m.def("meta_size", &lightllm::ops::meta_size, "Size (in bytes) of vllm::Signal metadata");
    m.def("group8_int8kv_flashdecoding_stage1", &group_int8kv_flashdecoding_attention, "INT8KV FLASHDECODING ATTENTION (CUDA)");
    m.def("group_int8kv_decode_attention", &group_int8kv_decode_attention, "INT8KV DECODE ATTENTION (CUDA)");
}

} // namespace ops
} // namespace lightllm