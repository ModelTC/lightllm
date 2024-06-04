import copy
import torch
import torch.functional as F
import torch.distributed as dist
from torch.nn.functional import layer_norm
import numpy as np
from typing import Tuple
from functools import partial

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.cohere.layer_weights.transformer_layer_weight import CohereTransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.cohere.triton_kernel.cohere_layernorm import layer_norm_fwd
from lightllm.models.cohere.triton_kernel.cohere_rotary_embed import cohere_rotary_emb_fwd
from lightllm.utils.infer_utils import mark_cost_time


class CohereTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_eps"]
        return

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        return input

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: CohereTransformerLayerWeight
    ) -> torch.Tensor:
        return input
    
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
        layer_norm_fwd(
            input_embdings.unsqueeze(1),
            layer_weight.att_norm_weight_.unsqueeze(0),
            self.eps_
        )
        attn_out = self._context_attention(input_embdings,
                                infer_state,
                                layer_weight=layer_weight)
        ffn_out = self._context_ffn(input_embdings, infer_state, layer_weight)
        input_embdings = residual + attn_out + ffn_out
        return input_embdings

    def token_forward(self, input_embdings, infer_state: InferStateInfo, layer_weight):
        residual = copy.deepcopy(input_embdings)
        layer_norm_fwd(
            input_embdings.unsqueeze(1),
            layer_weight.att_norm_weight_.unsqueeze(0),
            self.eps_
        )
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
        layer_norm_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            layer_weight.q_norm_,
            self.eps_,
        )
        layer_norm_fwd(
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            layer_weight.k_norm_,
            self.eps_,
        )
        cohere_rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return q, cache_kv
