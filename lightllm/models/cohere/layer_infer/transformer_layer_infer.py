import copy
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.utils.infer_utils import mark_cost_time


class CohereTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        return

    def _cohere_rotary_emb(self, x, cos, sin):
        dtype = x.dtype
        seq_len, h, dim = x.shape
        x = x.float()
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]
        rot_x = torch.stack([-x2, x1], dim=-1).flatten(-2)
        cos = cos.view((seq_len, 1, dim))
        sin = sin.view((seq_len, 1, dim))
        o = (x * cos) + (rot_x * sin)
        return o.to(dtype=dtype)

    def _qk_layernorm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer_weight: CohereTransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype

        q = q.to(torch.float32)
        k = k.to(torch.float32)

        q_mean = q.mean(-1, keepdim=True)
        k_mean = k.mean(-1, keepdim=True)

        q_variance = (q - q_mean).pow(2).mean(-1, keepdim=True)
        k_variance = (k - k_mean).pow(2).mean(-1, keepdim=True)

        q = (q - q_mean) * torch.rsqrt(q_variance + self.eps_)
        k = (k - k_mean) * torch.rsqrt(k_variance + self.eps_)
        
        q = layer_weight.q_norm_.to(torch.float32) * q
        k = layer_weight.k_norm_.to(torch.float32) * k

        return q.to(q_dtype), k.to(k_dtype)

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        return input

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        return input
    
    def _input_layer_norm(
        self,
        hidden_states,
        infer_state: LlamaInferStateInfo,
        layer_weight: CohereTransformerLayerWeight,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.eps_)
        hidden_states = layer_weight.att_norm_weight_.to(torch.float32) * hidden_states
        return hidden_states.to(input_dtype)
    
    def _bind_norm(self):
        self._att_norm = partial(CohereTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(CohereTransformerLayerInfer._ffn_norm, self)
        return
    
    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        return ffn_out.view(-1, self.embed_dim_)

    def _token_ffn(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        return ffn_out.view(-1, self.embed_dim_)

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv  = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        return o.view(-1, self.embed_dim_)

    # this impl dont to use @mark_cost_time
    def _token_attention(self, input_embding, infer_state: InferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        cache_kv = self._pre_cache_kv(infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, cache_kv, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        return o.view(-1, self.embed_dim_)

    def context_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        residual = copy.deepcopy(input_embdings)
        input_embdings = self._input_layer_norm(input_embdings, infer_state, layer_weight)
        attn_out = self._context_attention(input_embdings,
                                infer_state,
                                layer_weight=layer_weight)
        ffn_out = self._context_ffn(input_embdings, infer_state, layer_weight)
        input_embdings = residual + attn_out + ffn_out
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        residual = copy.deepcopy(input_embdings)
        input_embdings = self._input_layer_norm(input_embdings, infer_state, layer_weight)
        attn_out = self._token_attention(input_embdings,
                              infer_state,
                              layer_weight=layer_weight)
        ffn_out = self._token_ffn(input_embdings, infer_state, layer_weight)
        input_embdings = residual + attn_out + ffn_out
        return input_embdings

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        q, cache_kv[:, 0 : self.tp_k_head_num_, :] = self._qk_layernorm(
            q.view(-1, self.tp_q_head_num_, self.head_dim_), 
            cache_kv[:, 0 : self.tp_k_head_num_, :], 
            layer_weight,
        )

        q = self._cohere_rotary_emb(
            q.view(-1, self.tp_q_head_num_, self.head_dim_), 
            infer_state.position_cos,
            infer_state.position_sin,
        )
        cache_kv[:, 0 : self.tp_k_head_num_, :] = self._cohere_rotary_emb(
            cache_kv[:, 0 : self.tp_k_head_num_, :], 
            infer_state.position_cos,
            infer_state.position_sin,
        )
        
        return q, cache_kv
