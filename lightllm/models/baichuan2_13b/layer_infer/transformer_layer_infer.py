import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
import triton
from lightllm.models.baichuan13b.layer_weights.transformer_layer_weight import BaiChuan13bTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.models.baichuan2_13b.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.baichuan2_13b.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2
from lightllm.models.llama.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
import pdb

class Baichuan2_13bTransformerLayerInfer(LlamaTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return
    
    def _get_qkv(self, input, cache_k, cache_v, infer_state, layer_weight: BaiChuan13bTransformerLayerWeight) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_)
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.k_weight_,
                    out=cache_k.view(-1, self.tp_k_head_num_ * self.head_dim_))
        torch.mm(input.view(-1, self.embed_dim_), layer_weight.v_weight_,
                    out=cache_v.view(-1, self.tp_v_head_num_ * self.head_dim_))
        return q
    
    def _context_attention_kernel(self, q, k, v, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                              k.view(-1, self.tp_k_head_num_, self.head_dim_),
                              v.view(-1, self.tp_v_head_num_, self.head_dim_),
                              o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                              layer_weight.tp_alibi,
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        return o_tensor

    def _bind_func(self):
        """
        baichuan2_13b only support normal mode.  
        """
        self._token_attention_kernel = self._token_decode_attention_normal
        self._copy_kv_to_mem_cache = self._copy_kv_to_mem_cache_normal   
        return
    
    def _token_decode_attention_normal(self, q, infer_state: InferStateInfo, layer_weight):
        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_q_head_num_, self.head_dim_)
        att_m_tensor = torch.empty((self.tp_q_head_num_, total_token_num), dtype=q.dtype, device="cuda")
        # flash attention with alibi
        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      layer_weight.tp_alibi,
                      infer_state.req_manager.req_to_token_indexs,
                      infer_state.b_req_idx,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)
        
        if triton.__version__ == "2.0.0":
            prob = torch.empty_like(att_m_tensor)
            token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
            att_m_tensor = None

            o_tensor = torch.empty_like(q)

            token_att_fwd2(prob,
                        infer_state.mem_manager.value_buffer[self.layer_num_],
                        o_tensor.view(calcu_shape1),
                        infer_state.req_manager.req_to_token_indexs,
                        infer_state.b_req_idx,
                        infer_state.b_start_loc,
                        infer_state.b_seq_len)
            prob = None
            return o_tensor
        elif triton.__version__ >= "2.1.0":
            o_tensor = torch.empty_like(q)
            from lightllm.models.llama.triton_kernel.token_attention_softmax_and_reducev import token_softmax_reducev_fwd
            token_softmax_reducev_fwd(att_m_tensor, 
                                      infer_state.mem_manager.value_buffer[self.layer_num_],
                                      o_tensor.view(calcu_shape1),
                                      infer_state.req_manager.req_to_token_indexs,
                                      infer_state.b_req_idx,
                                      infer_state.b_start_loc,
                                      infer_state.b_seq_len,
                                      infer_state.other_kv_index)
            return o_tensor
        else:
            raise Exception("not support triton version")
