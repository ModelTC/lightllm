from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.functional as F
import triton

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_att1 import token_att_fwd, token_att_fwd_int8k
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2, token_att_fwd2_int8v
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.common.basemodel.quantize import matmul_quantize_int8, matmul_dequantize_int8, matmul_dequantize_int4
from lightllm.common.basemodel import TransformerLayerInferTpl
from lightllm.utils.infer_utils import mark_cost_time


class LlamaTransformerLayerInferINT8(LlamaTransformerLayerInfer):
    """
    Llama Model Inference using Triton W8/W4A16 kernel.
    When prefill, we use `matmul_dequantize_int8`, and use `matmul_quantize_int8` when decode.
    For better balance latency and accurcy.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)

    def _get_qkv(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight, matmul_int8_func) -> torch.Tensor:
        qkv_output = matmul_int8_func(input.view(-1, self.embed_dim_),
                                      layer_weight.qkv_fused_weight,
                                      layer_weight.qkv_fused_weight_scale)
        tp_hidden_dim = self.embed_dim_ // self.world_size_
        q = qkv_output[:, : tp_hidden_dim]
        k = qkv_output[:, tp_hidden_dim : tp_hidden_dim * 2]
        v = qkv_output[:, tp_hidden_dim * 2 :]
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_qkv_context(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_dequantize_int8)
    
    def _get_qkv_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        return self._get_qkv(input, infer_state, layer_weight, matmul_quantize_int8)

    def _get_o_context(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight) -> torch.Tensor:
        o_tensor = matmul_dequantize_int8(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_tensor

    def _get_o_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight) -> torch.Tensor:
        o_tensor = matmul_quantize_int8(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                        layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight, matmul_int8_func) -> torch.Tensor:
        gate_up_output = matmul_int8_func(input.view(-1, self.embed_dim_),
                                          layer_weight.gate_up_fused_weight,
                                          layer_weight.gate_up_fused_weight_scale)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        torch.nn.functional.silu(gate_up_output[:, 0], inplace=True)
        ffn1_out = gate_up_output[:, 0] * gate_up_output[:, 1]
        gate_up_output = None
        ffn2_out = matmul_dequantize_int8(ffn1_out, layer_weight.down_proj, layer_weight.down_proj_scale)
        ffn1_out = None
        return ffn2_out

    def _ffn_context(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_dequantize_int8)

    def _ffn_decode(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight) -> torch.Tensor:
        return self._ffn(input, infer_state, layer_weight, matmul_quantize_int8)

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_context(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_k, cache_v, infer_state, layer_weight)
        q = None
        o = self._get_o_context(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    @mark_cost_time("trans context ffn forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_context(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return

    def _token_attention(self, input_embding, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embding, infer_state, layer_weight)
        self._pre_cache_kv(infer_state, layer_weight)
        q, cache_k, cache_v = self._get_qkv_decode(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_k, cache_v, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o_decode(o, infer_state, layer_weight)
        if self.world_size_ > 1:
            dist.all_reduce(o, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o.view(-1, self.embed_dim_))
        return

    def _token_ffn(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight):
        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn_decode(input1, infer_state, layer_weight)
        input1 = None
        if self.world_size_ > 1:
            dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return
    
    def _pre_cache_kv(self, infer_state:LlamaInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight):
        '''
        We always do kv cache copy.
        '''
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            destindex_copy_kv(cache_k, infer_state.prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        else:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, mem_manager.value_buffer[self.layer_num_])


class LlamaTransformerLayerInferINT4(LlamaTransformerLayerInfer):
    """
    Llama Model Inference using Triton W4A16 kernel.
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)

    def _get_qkv(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight, matmul_int8_func) -> torch.Tensor:
        qkv_output = matmul_dequantize_int4(input.view(-1, self.embed_dim_),
                                            layer_weight.qkv_fused_weight,
                                            layer_weight.qkv_fused_weight_scale)
        tp_hidden_dim = self.embed_dim_ // self.world_size_
        q = qkv_output[:, : tp_hidden_dim]
        k = qkv_output[:, tp_hidden_dim : tp_hidden_dim * 2]
        v = qkv_output[:, tp_hidden_dim * 2 :]
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_o(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight) -> torch.Tensor:
        o_tensor = matmul_dequantize_int4(input.view(-1, self.tp_o_head_num_ * self.head_dim_),
                                          layer_weight.o_weight_, layer_weight.o_weight_scale_)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight:LlamaTransformerLayerWeight, matmul_int8_func) -> torch.Tensor:
        gate_up_output = matmul_dequantize_int4(input.view(-1, self.embed_dim_),
                                                layer_weight.gate_up_fused_weight,
                                                layer_weight.gate_up_fused_weight_scale)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        torch.nn.functional.silu(gate_up_output[:, 0], inplace=True)
        ffn1_out = gate_up_output[:, 0] * gate_up_output[:, 1]
        gate_up_output = None
        ffn2_out = matmul_dequantize_int4(ffn1_out, layer_weight.down_proj, layer_weight.down_proj_scale)
        ffn1_out = None
        return ffn2_out

    def _pre_cache_kv(self, infer_state:LlamaInferStateInfo, layer_weight)->Tuple[torch.Tensor, torch.Tensor]:
        '''
        Release kv buffer to save memory, since we allocate while kv projection.
        '''
        if infer_state.is_prefill:
            infer_state.prefill_key_buffer = None
            infer_state.prefill_value_buffer = None
        else:
            infer_state.decode_key_buffer = None
            infer_state.decode_value_buffer = None

    def _post_cache_kv(self, cache_k, cache_v, infer_state:LlamaInferStateInfo, layer_weight):
        '''
        We always do kv cache copy.
        '''
        mem_manager = infer_state.mem_manager
        if infer_state.is_prefill:
            destindex_copy_kv(cache_k, infer_state.prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        else:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, mem_manager.value_buffer[self.layer_num_])