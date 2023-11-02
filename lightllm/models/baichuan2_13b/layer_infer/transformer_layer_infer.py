import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.baichuan13b.layer_weights.transformer_layer_weight import BaiChuan13bTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.common.basemodel import InferStateInfo
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.models.baichuan2_13b.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.baichuan2_13b.triton_kernel.token_flashattention_nopad import token_attention_fwd

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
    
    def _token_attention_kernel(self, q, infer_state:InferStateInfo, layer_weight: BloomTransformerLayerWeight)->torch.Tensor:
        o_tensor = torch.empty_like(q)
        token_attention_fwd(q.view(-1, self.tp_q_head_num_, self.head_dim_),
                            infer_state.mem_manager.key_buffer[self.layer_num_],
                            infer_state.mem_manager.value_buffer[self.layer_num_],
                            o_tensor.view(-1, self.tp_q_head_num_, self.head_dim_),
                            layer_weight.tp_alibi,
                            infer_state.b_loc,
                            infer_state.b_start_loc,
                            infer_state.b_seq_len,
                            infer_state.max_len_in_batch)
        return o_tensor
