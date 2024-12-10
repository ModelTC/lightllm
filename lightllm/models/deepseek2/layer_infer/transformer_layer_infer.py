from typing import Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.triton_kernel.destindex_copy_kv import destindex_copy_kv
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
    context_attention_fwd_no_prompt_cache,
)
from lightllm.models.deepseek2.triton_kernel.context_flashattention_nopad_with_v import (
    context_attention_fwd_with_v,
    context_attention_fwd_no_prompt_cache_with_v,
)

from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.models.deepseek2.layer_infer.fused_moe import fused_experts, grouped_topk
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.silu_and_mul import silu_and_mul_fwd
from lightllm.models.chatglm2.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from functools import partial
from lightllm.models.llama.yarn_rotary_utils import get_deepseek_mscale
import os


class Deepseek2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(
        self, layer_num, tp_rank, world_size, network_config, mode=[], disable_qk_absorb=False, disable_vo_absorb=False, expert_parallel_mode="etp"
    ):
        self.tp_k_head_num_ = 1
        self.tp_v_head_num_ = 1
        self.qk_nope_head_dim = network_config["qk_nope_head_dim"]
        self.qk_rope_head_dim = network_config["qk_rope_head_dim"]
        self.q_lora_rank = network_config["q_lora_rank"]
        self.kv_lora_rank = network_config["kv_lora_rank"]
        self.expert_parallel_mode = expert_parallel_mode

        self.n_routed_experts = network_config["n_routed_experts"]

        self.is_moe = (
            network_config["n_routed_experts"] is not None
            and layer_num >= network_config["first_k_dense_replace"]
            and layer_num % network_config["moe_layer_freq"] == 0
        )
        self.disable_qk_absorb = disable_qk_absorb
        self.disable_vo_absorb = disable_vo_absorb

        self.n_shared_experts = network_config["n_shared_experts"]
        self.num_experts_per_tok = network_config["num_experts_per_tok"]
        self.norm_topk_prob = network_config["norm_topk_prob"]
        self.n_group = network_config["n_group"]
        self.topk_group = network_config["topk_group"]
        self.routed_scaling_factor = network_config["routed_scaling_factor"]

        self.softmax_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** (-0.5)
        if network_config.get("rope_scaling", None) is not None:
            self.rope_scaling = network_config["rope_scaling"]
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = get_deepseek_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale
        tp_split = True if expert_parallel_mode == "etp" else False
        super().__init__(layer_num, tp_rank, world_size, network_config, mode, tp_split)
        self.tp_o_head_num_ = self.tp_q_head_num_

        self.num_heads = network_config["num_attention_heads"]
        self.num_kv_heads = network_config["num_key_value_heads"]
        self.enable_opt_decoding_mha = os.getenv("ENABLE_OPT_DECODE_MHA", "False").upper() in ["ON", "TRUE", "1"]
        self.mla_type = "ACCM"

        return

    def _bind_attention(self):
        self._context_attention_kernel = partial(Deepseek2TransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(
            Deepseek2TransformerLayerInfer._token_gqa_decode_attention_flashdecoding, self
        )
        self._copy_kv_to_mem_cache = partial(Deepseek2TransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        if self.is_moe:
            if self.expert_parallel_mode == "etp":
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_etp, self)
            elif self.expert_parallel_mode == "edp":
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn_edp, self)
            else:
                self._ffn = partial(Deepseek2TransformerLayerInfer._moe_ffn, self)
        else:
            self._ffn = partial(LlamaTransformerLayerInfer._ffn, self)

    def _get_qkv(
        self,
        input: torch.Tensor,
        cache_kv,
        infer_state: LlamaInferStateInfo,
        layer_weight: Deepseek2TransformerLayerWeight,
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        if not self.disable_qk_absorb:  # ACC
            if self.q_lora_rank is None:
                q_nope = layer_weight.fuse_qk_weight_.mm(input)
                q_rope = layer_weight.q_rope_proj_.mm(input)
            else:
                q = layer_weight.q_a_proj_.mm(input)
                rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_, out=q)
                q_nope = layer_weight.fuse_qk_weight_.mm(q)
                q_rope = layer_weight.q_rope_proj_.mm(q)
            q_nope = q_nope.view(-1, self.tp_q_head_num_, self.kv_lora_rank)
            q_rope = q_rope.view(-1, self.tp_q_head_num_, self.qk_rope_head_dim)
        else:   # CC
            if self.q_lora_rank is None:
                q = layer_weight.q_weight_.mm(input)
            else:
                q = layer_weight.q_a_proj_.mm(input)
                rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_, out=q)
                q = layer_weight.q_b_proj_.mm(q)

            q = q.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim)
            q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
            if infer_state.use_dynamic_prompt_cache and infer_state.is_prefill:
                self.mla_type = "ACCM"
            else:
                self.mla_type = layer_weight.mla_type
            if self.mla_type == "ACCM":
                q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        layer_weight.kv_a_proj_with_mqa_.mm(input, out=cache_kv.view(-1, self.kv_lora_rank + self.qk_rope_head_dim))

        rmsnorm_forward(
            cache_kv[:, :, : self.kv_lora_rank],
            weight=layer_weight.kv_a_layernorm_.weight,
            eps=self.eps_,
            out=cache_kv[:, :, : self.kv_lora_rank],
        )

        rotary_emb_fwd(
            q_rope,
            cache_kv[:, :, self.kv_lora_rank :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        return (q_nope, q_rope), cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        if not self.disable_vo_absorb:
            input = input.view(-1, self.tp_q_head_num_ * self.kv_lora_rank)
            o_tensor = layer_weight.fuse_vo_weight_.mm(input)
        else:
            if self.mla_type == "ACCM":
                input = layer_weight.v_b_proj_.bmm(input.transpose(0, 1)).transpose(0, 1)
            o_tensor = layer_weight.o_weight_.mm(input.reshape(-1, self.tp_q_head_num_ * self.qk_nope_head_dim))
        return o_tensor

    def _CC_method(
        self, q, compressed_kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ):
        num_local_heads = self.num_heads
        num_local_kv_heads = self.num_kv_heads
        if self.world_size_ > 1 and self.expert_parallel_mode == "etp":
            num_local_heads //= self.world_size_
            num_local_kv_heads //= self.world_size_
        if infer_state.use_dynamic_prompt_cache:
            compressed_kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
        # CC
        compressed_kv, k_pe = torch.split(  # (b*s, 1, kv_lora + qk_r)
            compressed_kv, [layer_weight.kv_lora_rank, layer_weight.qk_rope_head_dim], dim=-1
        )
        compressed_kv = compressed_kv.view(-1, layer_weight.kv_lora_rank)
        k = self.alloc_tensor(
            [k_pe.shape[0], num_local_kv_heads, layer_weight.qk_nope_head_dim + layer_weight.qk_rope_head_dim],
            dtype=q[0].dtype,
        )
        k[..., layer_weight.qk_nope_head_dim :] = k_pe
        wk = layer_weight.k_b_proj_.weight.view(-1, layer_weight.k_b_proj_.weight.shape[-1])
        o_tensor = self.alloc_tensor([compressed_kv.shape[0], wk.shape[0]], dtype=q[0].dtype)
        torch.mm(compressed_kv, wk.transpose(0, 1), out=o_tensor)
        k[..., : layer_weight.qk_nope_head_dim] = o_tensor.view(-1, num_local_kv_heads, layer_weight.qk_nope_head_dim)
        trans_weight = layer_weight.v_b_proj_.weight.transpose(1, 2)
        wv = trans_weight.view(-1, trans_weight.shape[-1])
        o_tensor = self.alloc_tensor([compressed_kv.shape[0], wv.shape[0]], dtype=q[0].dtype)
        torch.mm(compressed_kv, wv.transpose(0, 1), out=o_tensor)
        v = o_tensor.view(-1, num_local_kv_heads, layer_weight.qk_nope_head_dim)
        return self._context_attention_kernel_with_v(q, k, v, infer_state, layer_weight)

    def _ACC_method(
        self, q, compressed_kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ):
        q_ne, q_pe = q
        num_local_heads = self.num_heads
        num_local_kv_heads = self.num_kv_heads
        if self.world_size_ > 1 and self.expert_parallel_mode == "etp":
            num_local_heads //= self.world_size_
            num_local_kv_heads //= self.world_size_
        # ACC
        q = self.alloc_tensor(
            [q_ne.shape[0], num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim], dtype=q_ne.dtype
        )
        q[..., self.kv_lora_rank :] = q_pe
        torch.bmm(  # TODO: 转换成einsum 或者 cublas
            q_ne.transpose(0, 1),  # (h, b*s, qk_n)
            layer_weight.k_b_proj_.weight,  # (h, qk_n, kv_lora)
            out=q[..., : self.kv_lora_rank].view(q_ne.shape[1], q_ne.shape[0], self.kv_lora_rank),
        ).transpose(
            0, 1
        )  # (b*s, h, kv_lora)
        q_nope, q_rope = torch.split(  # (b*s, h, qk_n + qk_r) -> (b*s, h, qk_n), (b*s, h, qk_r)
            q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        if self.enable_opt_decoding_mha:
            import lightllm_ppl_mla

            o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype)
            kvstarts = torch.cat(
                [infer_state.b_start_loc, infer_state.b_start_loc[-1:] + infer_state.b_seq_len[-1:]], dim=0
            )
            lightllm_ppl_mla.decode_mla(
                o_tensor,
                q,
                compressed_kv[: infer_state.mem_end, :, :],
                infer_state.b_start_loc,
                kvstarts,
                self.softmax_scale,
                q.shape[-1],
                q_nope.shape[-1],
            )
            output_parallel = o_tensor
        else:
            output_parallel = self._token_gqa_decode_attention_flashdecoding_origin(
                (q_nope, q_rope), infer_state, layer_weight
            )
        o_tensor = self.alloc_tensor(
            [output_parallel.shape[1], output_parallel.shape[0], self.qk_nope_head_dim], dtype=q_ne.dtype
        )
        torch.bmm(  # TODO: 转换成einsum 或者 cublas
            output_parallel.transpose(0, 1),  # (h, b*s, kv_lora)
            layer_weight.v_b_proj_.weight,  # (h, kv_lora, vo_d)
            out=o_tensor,
        ).transpose(
            0, 1
        )  # (b*s, h, vo_d)
        return o_tensor

    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ) -> torch.Tensor:
        if self.mla_type == "MIX":
            return self._context_attention_kernel_with_CC(q, kv, infer_state, layer_weight, out)
        else:
            return self._context_attention_kernel_origin(q, kv, infer_state, layer_weight, out)

    def _context_attention_kernel_with_CC(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight, out=None
    ) -> torch.Tensor:
        return self._CC_method(q, kv, infer_state, layer_weight)

    def _context_attention_kernel_with_v(
        self, q: Tuple[torch.Tensor, torch.Tensor], kv, v, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        nope_head_dim = q_nope.shape[-1]
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out
        if infer_state.use_dynamic_prompt_cache:
            context_attention_fwd_with_v(
                q_nope,
                q_rope,
                kv[:, :, :nope_head_dim],
                kv[:, :, nope_head_dim:],
                v,
                o_tensor.view(-1, self.tp_q_head_num_, nope_head_dim),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.softmax_scale,
            )
        else:
            context_attention_fwd_no_prompt_cache_with_v(
                q_nope,
                q_rope,
                kv[:, :, :nope_head_dim],
                kv[:, :, nope_head_dim:],
                v,
                o_tensor.view(-1, self.tp_q_head_num_, nope_head_dim),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )
        q_nope = None
        q_rope = None
        return o_tensor

    def _context_attention_kernel_origin(
        self, q: Tuple[torch.Tensor, torch.Tensor], kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        q_nope, q_rope = q
        o_tensor = self.alloc_tensor(q_nope.shape, dtype=q_nope.dtype) if out is None else out

        if infer_state.use_dynamic_prompt_cache:
            kv = infer_state.mem_manager.kv_buffer[self.layer_num_]
            context_attention_fwd(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.b_ready_cache_len,
                infer_state.max_len_in_batch,
                infer_state.req_manager.req_to_token_indexs,
                self.softmax_scale,
            )
        else:
            context_attention_fwd_no_prompt_cache(
                q_nope,
                q_rope,
                kv[:, :, : self.kv_lora_rank],
                kv[:, :, self.kv_lora_rank :],
                o_tensor.view(-1, self.tp_q_head_num_, self.kv_lora_rank),
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.max_len_in_batch,
                self.softmax_scale,
            )
        q_nope = None
        q_rope = None
        return o_tensor

    def _token_gqa_decode_attention_flashdecoding(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        if self.mla_type == "MIX":
            return self._token_gqa_decode_attention_flashdecoding_with_ACC(q, infer_state, layer_weight, out)
        else:
            return self._token_gqa_decode_attention_flashdecoding_origin(q, infer_state, layer_weight, out)

    def _token_gqa_decode_attention_flashdecoding_with_ACC(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        compressed_kv = infer_state.mem_manager.kv_buffer[self.layer_num_][: infer_state.mem_end, :, :]
        return self._ACC_method(q, compressed_kv, infer_state, layer_weight)

    def _token_gqa_decode_attention_flashdecoding_origin(
        self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ):
        q_nope, q_rope = q
        kv = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank]
        kv_rope = infer_state.mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :]
        return gqa_token_decode_attention_flash_decoding(
            q_nope,
            q_rope,
            kv,
            kv_rope,
            infer_state,
            self.tp_q_head_num_,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.qk_nope_head_dim,
            self.softmax_scale,
            alloc_tensor_func=self.alloc_tensor,
        )

    def _copy_kv_to_mem_cache_normal(self, buffer, mem_index, mem_manager):
        destindex_copy_kv(
            buffer[:, :, : self.kv_lora_rank],
            buffer[:, :, self.kv_lora_rank :],
            mem_index,
            mem_manager.kv_buffer[self.layer_num_][:, :, : self.kv_lora_rank],
            mem_manager.kv_buffer[self.layer_num_][:, :, self.kv_lora_rank :],
        )
        return

    def _moe_ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)

        router_logits = layer_weight.moe_gate.mm(hidden_states)
        layer_weight.experts.experts(
            hidden_states,
            router_logits=router_logits,
            top_k=self.num_experts_per_tok,
            renormalize=self.norm_topk_prob,
            use_grouped_topk=self.n_group,
            topk_group=self.topk_group,
            num_expert_group=self.n_group,
        )

        hidden_states.mul_(self.routed_scaling_factor)

        if self.n_shared_experts is not None:
            hidden_states.add_(shared_output)

        return hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_etp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        world_size_ = self.world_size_
        # num_local_experts = self.n_shared_experts // world_size_
        # local_expert_offset = self.tp_rank_ * num_local_experts
        num_experts_per_token = self.num_experts_per_tok
        num_experts = self.n_routed_experts
        # num_expert_groups = self.n_group
        # num_groups_per_token = self.topk_group
        gating_scaling_factor = self.routed_scaling_factor
        # gating_normalize_prob = self.norm_topk_prob
        rank_self = self.tp_rank_

        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        final_hidden_states = torch.empty(
            num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # router_logits_len = hidden_states.shape[0]*layer_weight.moe_gate.shape[1]
        router_logits = layer_weight.moe_gate.mm(hidden_states)

        # now some parameter is not supported yet
        # assert gating_normalize_prob is False
        # assert num_expert_groups<=1

        import lightllm_moe_etp_kernel

        lightllm_moe_etp_kernel.moe_fused_all(
            router_logits.contiguous(),
            hidden_states.contiguous(),
            layer_weight.gate_up_proj.weight.contiguous(),  # transpose
            layer_weight.down_proj.weight.contiguous(),  # transpose
            layer_weight.experts.expert_gate_up_proj_etp.contiguous(),
            layer_weight.experts.expert_down_proj_etp.contiguous(),
            infer_state.mem_manager.work_buffer.contiguous(),
            infer_state.mem_manager.work_buffer.nelement(),
            final_hidden_states.contiguous(),
            rank_self,
            gating_scaling_factor,
            num_experts,
            num_experts_per_token,
            num_tokens,
            world_size_,
            True,
            hidden_dim,
            layer_weight.gate_up_proj.weight.size(1) // 2,
            layer_weight.experts.expert_gate_up_proj_etp.size(1) // 2,
            self.n_shared_experts is not None,
        )

        router_logits = None

        return final_hidden_states.view(num_tokens, hidden_dim)

    def _moe_ffn_edp(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Deepseek2TransformerLayerWeight
    ) -> torch.Tensor:
        world_size = self.world_size_        

        num_experts_per_token = self.num_experts_per_tok
        num_experts = self.n_routed_experts
        num_local_experts = num_experts // world_size
        # num_expert_groups = self.n_group
        # num_groups_per_token = self.topk_group
        gating_scaling_factor = self.routed_scaling_factor
        # gating_normalize_prob = self.norm_topk_prob
        rank_self = self.tp_rank_
        hidden_states = input.view(-1, self.embed_dim_)
        num_tokens, hidden_dim = hidden_states.shape

        if self.n_shared_experts is not None:
            shared_output = LlamaTransformerLayerInfer._ffn(self, hidden_states, infer_state, layer_weight)
        # final_hidden_states = torch.empty(
        #     num_tokens, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype
        # )

        router_logits = layer_weight.moe_gate.mm(hidden_states)

        states_expand_permute, expert_weights, invert_permutation, expert_offset = moe_select(
            hidden_states, router_logits, num_experts, num_experts_per_token,
            1, 1,
            gating_scaling_factor, False,
            gating_method="greedy"
        )
        device = hidden_states.device
        flat_stats = states_expand_permute.view(-1, hidden_dim)
        global_exp_local_token = torch.tensor([expert_offset[i+1] - expert_offset[i] for i in range(num_experts)], dtype=torch.int64, device=device)    # [num_experts]
        
        local_exp_global_token = torch.zeros(num_experts, dtype=torch.int64, device=device)  # [rano0_local_exp, rank1_local_exp, ...]
        dist.all_to_all_single(local_exp_global_token, global_exp_local_token)
        
        input_splits = global_exp_local_token.reshape(world_size, num_local_experts).sum(dim=1) # [world_size]
        output_splists = local_exp_global_token.reshape(world_size, num_local_experts).sum(dim=1)  # [world_size]

        local_exp_global_input = torch.zeros((output_splists.sum().item(), hidden_dim), dtype=hidden_states.dtype, device=device)
        dist.all_to_all_single(local_exp_global_input, flat_stats, output_split_sizes=output_splists.tolist(), input_split_sizes=input_splits.tolist())

        input_chunk_idxs = torch.arange(num_experts)
        sorted_local_exp_index = input_chunk_idxs.reshape(world_size, num_local_experts).T.ravel()
        restore_local_exp_index = input_chunk_idxs.reshape(num_local_experts, world_size).T.ravel()

        # sort chunk by idx
        expert_sorted_token = local_exp_global_token.reshape(world_size, -1).sum(dim=0) # [num_local_experts]
        sorted_local_exp_global_token = local_exp_global_token.reshape(world_size, -1).transpose(0, 1).contiguous().view(-1)

        def permute_chunks_by_idxs(input: torch.Tensor, split_size: torch.Tensor, sorted_idxs: torch.Tensor):
            """
                sort chunks by idx, 
            """
            splited_input = input.split(split_size.tolist())
            output = torch.cat([splited_input[i] for i in sorted_idxs.tolist()], dim=0)
            return output

        expert_sorted_input = permute_chunks_by_idxs(local_exp_global_input, local_exp_global_token, sorted_local_exp_index)
        expert_sorted_token_offset = [0]
        # new offset
        for i in range(num_local_experts):
            expert_sorted_token_offset.append(expert_sorted_token_offset[i] + expert_sorted_token[i])
        
        down_proj_output = torch.zeros_like(local_exp_global_input)
        for i in range(num_local_experts):
            token_beg_idx = expert_sorted_token_offset[i]
            token_end_idx = expert_sorted_token_offset[i+1]
            if token_beg_idx == token_end_idx:
                continue
            local_expert_idx = i
            up_proj_output = F.linear(
                        expert_sorted_input[token_beg_idx:token_end_idx],
                        layer_weight.experts.expert_gate_up_proj_etp[local_expert_idx],
            )
            # up_proj_output = act_fn(up_proj_output)
            act_output = torch.empty(up_proj_output.shape[0], up_proj_output.shape[1] // 2, device=up_proj_output.device, dtype=up_proj_output.dtype)

            silu_and_mul_fwd(up_proj_output, act_output)

            down_proj_output[token_beg_idx : token_end_idx] = F.linear(
                act_output, 
                layer_weight.experts.expert_down_proj_etp[local_expert_idx],
            )

        # restore chunks
        restore_down_proj_output = permute_chunks_by_idxs(down_proj_output, sorted_local_exp_global_token, restore_local_exp_index)    
        input_splits2 = output_splists
        output_splits2 = input_splits

        global_exp_local_output = torch.zeros_like(flat_stats, dtype=hidden_states.dtype, device=device)
        dist.all_to_all_single(global_exp_local_output, restore_down_proj_output, output_split_sizes=output_splits2.tolist(), input_split_sizes=input_splits2.tolist())

        final_hidden_states = moe_reduce(global_exp_local_output, expert_weights, invert_permutation, num_experts_per_token)
        if shared_output is not None:
            final_hidden_states.add_(shared_output)
        # now some parameter is not supported yet
        # assert gating_normalize_prob is False
        # assert num_expert_groups<=1
        return final_hidden_states

def moe_select(X: torch.Tensor, scores: torch.Tensor,
                num_experts: int, num_experts_per_token: int,
                num_expert_groups: int = 1, num_groups_per_token: int = 1,
                gating_scaling_factor: float = 1.0,
                gating_normalize_prob: bool = False,
                gating_method: str='greedy'):
    origin_shape = X.shape
    X_expand_permute = X.view(-1, X.shape[-1])
    _scores = scores.softmax(dim=-1, dtype=torch.float32).view(-1, num_experts).type_as(scores)
    if 'greedy' in gating_method and len('greedy') == len(gating_method):
        expert_weights, expert_indices = torch.topk(_scores, num_experts_per_token, dim=-1)
    elif 'grouped_limited_greedy' in gating_method and len('grouped_limited_greedy') == len(gating_method):
        group_scores = (
            _scores.view(_scores.shape[0], num_expert_groups, -1).max(dim=-1).values
        )  # [n, n_group]
        group_idx = torch.topk(
            group_scores, k=num_groups_per_token, dim=-1, sorted=False
        )[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(_scores.shape[0], num_expert_groups, num_experts_per_token // num_expert_groups
            )
            .reshape(_scores.shape[0], -1)
        )  # [n, e]
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        expert_weights, expert_indices = torch.topk(tmp_scores, num_experts_per_token, dim=-1)

    if num_experts_per_token > 1 and gating_normalize_prob:
        denominator = expert_weights.sum(dim=-1, keepdim=True) + 1e-20
        expert_weights = expert_weights / denominator
    else:
        expert_weights *= gating_scaling_factor

    flat_expert_indices = expert_indices.view(-1)   # (seqlen * num_experts_per_token)
    
    sorted_expert_indices, permute_token_idx = flat_expert_indices.sort(stable=True)
    X_expand_permute = X_expand_permute.repeat_interleave(num_experts_per_token, dim=0) # (seqlen * num_experts_per_token, hidden_dim)
    
    X_expand_permute = X_expand_permute[permute_token_idx]

    invert_permutation = torch.full_like(permute_token_idx, -1, device=scores.device)
    for i in range(len(permute_token_idx)):
        reidx = permute_token_idx[i]
        invert_permutation[reidx] = i

    expert_offset = torch.full((num_experts + 1,), -1, device=scores.device)
    ptr = 0
    for i in range(num_experts):
        while(ptr < len(sorted_expert_indices) and sorted_expert_indices[ptr] < i):
            ptr += 1
        expert_offset[i] = ptr
    expert_offset[num_experts] = X_expand_permute.size(0)
    X_expand_permute = X_expand_permute.view(*origin_shape[:-1], num_experts_per_token, -1)
    invert_permutation = invert_permutation.view(*origin_shape[:-1], num_experts_per_token)

    return X_expand_permute, expert_weights, invert_permutation, expert_offset 


def moe_reduce(Y: torch.Tensor, expert_weights: torch.Tensor,
            invert_permutation: torch.Tensor, num_experts_per_token: int):
    Y = Y.view(-1, Y.shape[-1])
    Y_out = Y[invert_permutation].view(*expert_weights.shape, -1)
    Y_out = (Y_out * expert_weights.unsqueeze(-1)).sum(dim=-2) # [*, hidden_dim]
    return Y_out