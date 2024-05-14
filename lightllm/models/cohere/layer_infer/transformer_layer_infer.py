import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd


class CohereTransformerLayerInfer(LlamaTransformerLayerInfer):
    def _bind_norm(self):
        self._att_norm = partial(CohereTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(CohereTransformerLayerInfer._ffn_norm, self)
        return

    def _att_norm(
        self,
        hidden_states,
        infer_state: LlamaInferStateInfo,
        layer_weight: CohereTransformerLayerWeight,
    ):
        return hidden_states
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.eps_)
        hidden_states = layer_weight.att_norm_weight_.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)
    
    def _cohere_layernorm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer_weight: CohereTransformerLayerWeight,
        eps=1e-5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        q_mean = q.mean(-1, keepdim=True)
        k_mean = k.mean(-1, keepdim=True)

        q_variance = (q - q_mean).pow(2).mean(-1, keepdim=True)
        k_variance = (k - k_mean).pow(2).mean(-1, keepdim=True)

        q = (q - q_mean) * torch.rsqrt(q_variance + eps)
        k = (k - k_mean) * torch.rsqrt(k_variance + eps)
        
        q = layer_weight.q_norm_.to(torch.float32).view(-1) * q
        k = layer_weight.k_norm_.to(torch.float32).view(-1) * k

        k = k.to(k_dtype)
        return q.to(q_dtype)

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        return input

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        #q = self._cohere_layernorm(q, cache_kv[:, 0 : self.tp_k_head_num_, :], layer_weight)
        return q, cache_kv
