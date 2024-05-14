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

        q = layer_weight.q_norm_.to(torch.float32) * q
        k = layer_weight.k_norm_.to(torch.float32) * k

        return q.to(q_dtype), k.to(k_dtype)

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        pass

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
        return self._cohere_layernorm(q, cache_kv, layer_weight)
