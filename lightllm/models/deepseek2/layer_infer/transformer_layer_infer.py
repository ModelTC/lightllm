import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache
)
from lightllm.models.deepseek2.triton_kernel.flash_decoding import token_decode_attention_flash_decoding
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from functools import partial


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.qk_nope_head_dim = self.config["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.config["qk_rope_head_dim"]
        self.q_lora_rank = self.config["q_lora_rank"]
        self.kv_lora_rank = self.config["kv_lora_rank"]
        self._bind_func()
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = partial(Deepseek2TransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(
            Deepseek2TransformerLayerInfer._token_decode_attention_flashdecoding, self
        )
        self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)

    def _get_qkv(
        self, input, cache_kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if self.q_lora_rank is None:
            q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        else:
            q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_a_proj_)
            q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_, eps=self.eps_)
            q = torch.mm(q, layer_weight.q_b_proj_)
        
        q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_nope = torch.matmul(q_nope.unsqueeze(2), layer_weight.k_b_proj_).squeeze(2)
        
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_a_proj_with_mqa_,
            out=cache_kv.view(-1, self.kv_lora_rank + self.qk_rope_head_dim),
        )
        cache_kv[:, :, : self.kv_lora_rank] = rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_, 
            eps=self.eps_
        )
        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank:],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return (q_nope, q_rope), cache_kv

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = torch.empty_like(q_nope) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q_nope,
                q_rope,
                kv[:, : , : self.kv_lora_rank],
                kv[:, : , self.kv_lora_rank: ],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.qk_nope_head_dim
            )
        else:
            context_attention_fwd_no_prompt_cache(
                q_nope,
                q_rope,
                kv[:, : , : self.kv_lora_rank],
                kv[:, : , self.kv_lora_rank: ],
                o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.qk_nope_head_dim
            )

        return o_tensor
    
    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        o_tensor = torch.mm(input.view(-1, self.tp_o_head_num_ * self.head_dim_), layer_weight.o_weight_)
        return o_tensor
    
    def _token_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        q_nope, q_rope = q
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank]
        kv_rope = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank: ]
        return token_decode_attention_flash_decoding(
            q_nope, q_rope, kv, kv_rope, infer_state, self.tp_q_head_num_, self.kv_lora_rank, self.qk_rope_head_dim, self.qk_nope_head_dim
        )
    
    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(
            buffer[:, :, : self.kv_lora_rank],
            buffer[:, :, self.kv_lora_rank :],
            mem_index,
            mem_manager.kv_buffer[self.layer_num_][ :, :, : self.kv_lora_rank],
            mem_manager.kv_buffer[self.layer_num_][ :, :, self.kv_lora_rank:],
        )
        return