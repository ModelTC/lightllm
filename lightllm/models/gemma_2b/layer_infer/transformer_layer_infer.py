import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from typing import Tuple
from functools import partial
import triton

from lightllm.models.gemma_2b.layer_weights.transformer_layer_weight import Gemma_2bTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.gemma_2b.triton_kernel.gelu_and_mul import gelu_and_mul_fwd
from lightllm.models.gemma_2b.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.gemma_2b.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.gemma_2b.triton_kernel.rmsnorm import rmsnorm_forward

from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Gemma_2bTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_k_head_num_ = network_config["num_key_value_heads"] # [SYM] always == 1
        self.tp_v_head_num_ = network_config["num_key_value_heads"]
        return
    
    def _bind_attention(self):
        self._context_attention_kernel = partial(Gemma_2bTransformerLayerInfer._context_attention_kernel, self)
        self._token_attention_kernel = partial(Gemma_2bTransformerLayerInfer._token_decode_attention_normal, self)
        self._copy_kv_to_mem_cache = partial(LlamaTransformerLayerInfer._copy_kv_to_mem_cache_normal, self)
        return
    
    def _bind_norm(self):
        self._att_norm = partial(Gemma_2bTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(Gemma_2bTransformerLayerInfer._ffn_norm, self)
        return

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bTransformerLayerWeight
    ) -> torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.att_norm_weight_, eps=self.eps_)

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bTransformerLayerWeight
    ) -> torch.Tensor:
        return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)
    
    def _context_attention_kernel(
        self, q, kv, infer_state: LlamaInferStateInfo, layer_weight, out=None
    ) -> torch.Tensor:
        o_tensor = torch.empty_like(q) if out is None else out
        context_attention_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            kv[:, 0 : self.tp_k_head_num_, :],
            kv[:, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :],
            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )
        return o_tensor
    
    def _token_decode_attention_normal(self, q, infer_state: LlamaInferStateInfo, layer_weight, out=None):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)

        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(
            q.view(calcu_shape1),
            infer_state.mem_manager.kv_buffer[self.layer_num_][:, 0 : self.tp_k_head_num_, :],
            att_m_tensor,
            infer_state.req_manager.req_to_token_indexs,
            infer_state.b_req_idx,
            infer_state.b_start_loc,
            infer_state.b_seq_len,
            infer_state.max_len_in_batch,
        )

        o_tensor = torch.empty_like(q) if out is None else out

        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            token_softmax_fwd(
                att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch
            )
            att_m_tensor = None
            token_att_fwd2(
                prob,
                infer_state.mem_manager.kv_buffer[self.layer_num_][
                    :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
                ],
                o_tensor.view(calcu_shape1),
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
            )
            prob = None
            return o_tensor
        elif triton.__version__ >= "2.1.0":
            from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import (
                token_softmax_reducev_fwd,
            )

            token_softmax_reducev_fwd(
                att_m_tensor,
                infer_state.mem_manager.kv_buffer[self.layer_num_][
                    :, self.tp_k_head_num_ : self.tp_k_head_num_ + self.tp_v_head_num_, :
                ],
                o_tensor.view(calcu_shape1),
                infer_state.req_manager.req_to_token_indexs,
                infer_state.b_req_idx,
                infer_state.b_start_loc,
                infer_state.b_seq_len,
                infer_state.other_kv_index,
            )
            return o_tensor
        else:
            raise Exception("not support triton version")

    def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bTransformerLayerWeight) -> torch.Tensor:
        up_gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
        ffn1_out = gelu_and_mul_fwd(up_gate_out)
        input = None
        up_gate_out = None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out

    # def _ffn(self, input, infer_state: LlamaInferStateInfo, layer_weight: Gemma_2bTransformerLayerWeight) -> torch.Tensor:
    #     gate_up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_up_proj)
    #     size = gate_up_out.shape[1]
    #     gate_out, up_out = gate_up_out[:, 0: size // 2], gate_up_out[:, size // 2:]
    #     gate_out = torch.nn.functional.gelu(gate_out)
    #     gate_out.mul_(up_out)
    #     input = None
    #     ffn2_out = torch.mm(gate_out, layer_weight.down_proj)
    #     gate_out, up_out = None, None
    #     return ffn2_out