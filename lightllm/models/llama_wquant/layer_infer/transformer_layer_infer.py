from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.functional as F
import triton
from functools import partial

from lightllm.models.llama_wquant.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeightQuantized
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel import TransformerLayerInferWeightQuantTpl
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import matmul_quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int8 import matmul_dequantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import matmul_dequantize_int4_s1, matmul_dequantize_int4_s2, matmul_dequantize_int4_gptq
from lightllm.utils.infer_utils import mark_cost_time

 
class LlamaTransformerLayerInferWquant(TransformerLayerInferWeightQuantTpl):
    """
    """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.tp_k_head_num_ = self.tp_q_head_num_
        self.tp_v_head_num_ = self.tp_q_head_num_
        self.tp_o_head_num_ = self.tp_q_head_num_
        self.head_dim_ = network_config["hidden_size"] // network_config["num_attention_heads"]
        self.embed_dim_ = network_config["hidden_size"]
        
        self.inter_dim_ = network_config['intermediate_size']
        self._bind_func()
        return
    
    def _bind_func(self):
        self._att_norm = partial(LlamaTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(LlamaTransformerLayerInfer._ffn_norm, self)

        if "int8weight" in self.mode:
            self._wquant_matmul_for_qkv = self._wquant_matmul_triton_int8weight_only_quant
            self._wquant_matmul_for_o = self._wquant_matmul_triton_int8weight_only_quant
            self._wquant_matmul_for_ffn_up = self._wquant_matmul_triton_int8weight_only_quant
            self._wquant_matmul_for_ffn_down = self._wquant_matmul_triton_int8weight_only_quant
        elif "int4weight" in self.mode:
            self._wquant_matmul_for_qkv = self._wquant_matmul_triton_int4weight_only_quant
            self._wquant_matmul_for_o = self._wquant_matmul_triton_int4weight_only_quant
            self._wquant_matmul_for_ffn_up = self._wquant_matmul_triton_int4weight_only_quant
            self._wquant_matmul_for_ffn_down = self._wquant_matmul_triton_int4weight_only_quant

        else:
            raise Exception(f"error mode {self.mode}")
        
        self._bind_attention()
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = partial(LlamaTransformerLayerInfer._context_attention_kernel, self)
        if "ppl" in self.mode and "int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_ppl_int8kv, self)
        elif "int8kv" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_int8kv, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_int8kv, self)
        elif "flashdecoding" in self.mode:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_flashdecoding, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)   
        else:
            self._token_attention_kernel = partial(LlamaTransformerLayerInfer._token_decode_attention_normal, self)
            self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        return

    def _get_qkv(self, input, cache_k, cache_v, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized):
        qkv_output = self._wquant_matmul_for_qkv(input.view(-1, self.embed_dim_), 
                                                    quant_weight_params=layer_weight.qkv_weight_,
                                                    is_prefill=infer_state.is_prefill)

        tp_hidden_dim = self.embed_dim_ // self.world_size_
        q = qkv_output[:, : tp_hidden_dim]
        k = qkv_output[:, tp_hidden_dim : tp_hidden_dim * 2]
        v = qkv_output[:, tp_hidden_dim * 2 :]
        rotary_emb_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_), infer_state.position_cos, infer_state.position_sin)
        cache_k_ = k.view(-1, self.tp_k_head_num_, self.head_dim_)
        rotary_emb_fwd(cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_v_ = v.view(-1, self.tp_v_head_num_, self.head_dim_)
        return q, cache_k_, cache_v_

    def _get_o(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        o_tensor = self._wquant_matmul_for_o(input, 
                                             quant_weight_params=layer_weight.o_weight_,
                                             is_prefill=infer_state.is_prefill)
        return o_tensor

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeightQuantized) -> torch.Tensor:
        gate_up_output = self._wquant_matmul_for_ffn_up(input.view(-1, self.embed_dim_),
                                                        quant_weight_params=layer_weight.gate_up_proj,
                                                        is_prefill=infer_state.is_prefill)
        input = None
        tp_inter_dim = self.inter_dim_ // self.world_size_
        gate_up_output = gate_up_output.view(-1, 2, tp_inter_dim)
        torch.nn.functional.silu(gate_up_output[:, 0], inplace=True)
        ffn1_out = gate_up_output[:, 0] * gate_up_output[:, 1]
        gate_up_output = None
        ffn2_out = self._wquant_matmul_for_ffn_down(ffn1_out, 
                                                    quant_weight_params=layer_weight.down_proj,
                                                    is_prefill=infer_state.is_prefill)
        ffn1_out = None
        return ffn2_out
    
    def _wquant_matmul_triton_int8weight_only_quant(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        assert bias is None and has_act == False
        if is_prefill:
            qweight, scale = quant_weight_params
            return matmul_dequantize_int8(input, qweight, scale, out=out)
        else:
            qweight, scale = quant_weight_params
            return matmul_quantize_int8(input, qweight, scale, out=out)
        
    def _wquant_matmul_triton_int4weight_only_quant(self, input, quant_weight_params, is_prefill, out=None, bias=None, has_act=False):
        assert bias is None and has_act == False
        if is_prefill:
            qweight, scale, zeros, int4_q_group_size = quant_weight_params
            return matmul_dequantize_int4_s1(input, qweight, scale, zeros, int4_q_group_size, out=out)
        else:
            qweight, scale, zeros, int4_q_group_size = quant_weight_params
            return matmul_dequantize_int4_gptq(input, qweight, scale, zeros, int4_q_group_size, output=out)