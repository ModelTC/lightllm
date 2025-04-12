import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.cohere.layer_infer.transformer_layer_infer import CohereTransformerLayerInfer
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward
from lightllm.models.gemma3.layer_weights.transformer_layer_weight import Gemma3TransformerLayerWeight
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd

from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


class Gemma3TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        self.tp_k_head_num_ = network_config["num_key_value_heads"]  # [SYM] always == 1
        self.tp_v_head_num_ = network_config["num_key_value_heads"]
        self.eps_ = 1e-6
        self.head_dim_ = 256
        return

    def _pre_feedforward_layernorm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        return layernorm_forward(
            input.unsqueeze(1), layer_weight.pre_feedforward_layernorm_weight_.weight.unsqueeze(0), self.eps_
        ).squeeze(1)

    def _post_feedforward_layernorm(self, input, infer_state, layer_weight: Gemma3TransformerLayerWeight):
        return layernorm_forward(
            input.unsqueeze(1), layer_weight.post_feedforward_layernorm_weight_.weight.unsqueeze(0), self.eps_
        ).squeeze(1)

    def _bind_norm(self):
        super()._bind_norm()
        self._q_norm = partial(CohereTransformerLayerInfer._q_norm, self)
        self._k_norm = partial(CohereTransformerLayerInfer._k_norm, self)        
        self._pre_feedforward_layernorm = partial(Gemma3TransformerLayerInfer._pre_feedforward_layernorm, self)
        self._post_feedforward_layernorm = partial(Gemma3TransformerLayerInfer._post_feedforward_layernorm, self)

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight
    ) -> torch.Tensor:
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(
            input, out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_)
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)

        # gemma3 use qk norm
        q = q.view(-1, self.tp_q_head_num_, self.head_dim_)
        k = cache_kv[:, 0 : self.tp_k_head_num_, :]
        q = self._q_norm(q, infer_state, layer_weight)
        cache_kv[:, 0 : self.tp_k_head_num_, :] = self._k_norm(k, infer_state, layer_weight)

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv


    def _ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma3TransformerLayerWeight
    ) -> torch.Tensor:
        input = self._pre_feedforward_layernorm(input, infer_state, layer_weight)
        up_gate_out = layer_weight.gate_up_proj.mm(input.view(-1, self.embed_dim_))
        ffn1_out = self.alloc_tensor((input.size(0), up_gate_out.size(1) // 2), input.dtype)
        gelu_and_mul_fwd(up_gate_out, ffn1_out)
        input = None
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(
            ffn1_out,
        )
        ffn1_out = None
        ffn2_out = self._post_feedforward_layernorm(ffn2_out, infer_state, layer_weight)
        return ffn2_out
