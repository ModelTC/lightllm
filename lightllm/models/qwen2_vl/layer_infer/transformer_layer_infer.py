import torch
import triton
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
    context_attention_fwd_ppl_int8kv,
)
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.models.llama.triton_kernel.ppl_quant_copy_kv import destindex_copy_dequantize_kv
from lightllm.distributed.communication_op import all_gather_into_tensor, reduce_scatter_tensor
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.qwen2_vl.infer_struct import Qwen2VLInferStateInfo
from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2VLTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode):
        super().__init__(layer_num, network_config, mode)
        self.rope_scaling = network_config["rope_scaling"]
        return

    def _get_qkv(
        self, input, cache_kv, infer_state: Qwen2VLInferStateInfo, layer_weight: Qwen2TransformerLayerWeight
    ) -> torch.Tensor:
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        apply_multimodal_rotary_pos_emb(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,  #
            infer_state.position_sin,  #
            self.rope_scaling["mrope_section"],
        )
        return q, cache_kv
