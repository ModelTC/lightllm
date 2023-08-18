import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.utils.infer_utils import mark_cost_time
from lightllm.models.starcoder.layer_infer.infer_struct import StarcoderInferStateInfo
from lightllm.models.bloom.layer_infer.transformer_layer_inference import BloomTransformerLayerInfer
from lightllm.models.starcoder.layer_weights.transformer_layer_weight import StarcoderTransformerLayerWeight

from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv, destindex_copy_quantize_kv
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.llama2.triton_kernel.context_flashattention_nopad import context_attention_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_att1 import token_att_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_softmax import token_softmax_fwd
from lightllm.models.llama2.triton_kernel.token_attention_nopad_reduceV import token_att_fwd2


class StarcoderTransformerLayerInfer(BloomTransformerLayerInfer):
    """
    """
    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=""):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        self.tp_key_head_num_ = network_config["num_hidden_layers"] 
        self.tp_value_head_num_ = network_config["num_hidden_layers"] 
        self.key_value_head_num_ = 1
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.tp_head_sum_dim_ = self.tp_head_num_ * self.head_dim_
        
        self.tp_kv_head_num = 1
        self.tp_kv_head_sum_dim_ = self.tp_kv_head_num * self.head_dim_
        return

    @mark_cost_time("trans context flash forward time cost")  # dont to remove this, will make performence down, did not know why
    def _context_flash_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeight):
        mem_manager = infer_state.mem_manager
        prefill_mem_index = infer_state.prefill_mem_index
        prefill_key_buffer = infer_state.prefill_key_buffer
        prefill_value_buffer = infer_state.prefill_value_buffer
        
        total_token_num = infer_state.total_token_num
        calcu_shape1 = (total_token_num, self.tp_head_num_, self.head_dim_)
        input1 = layernorm_forward(input_embding, weight=layer_weight.input_layernorm_weight_, bias=layer_weight.input_layernorm_bias_, eps=self.layer_norm_eps_)
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=prefill_key_buffer[0:total_token_num, :, :].view(-1, self.tp_kv_head_sum_dim_))
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=prefill_value_buffer[0:total_token_num, :, :].view(-1, self.tp_kv_head_sum_dim_))
        
        input1 = None
        o_tensor = torch.empty_like(q)
        context_attention_fwd(q.view(calcu_shape1),
                              prefill_key_buffer[0:total_token_num, :, :],
                              prefill_value_buffer[0:total_token_num, :, :],
                              o_tensor.view(calcu_shape1),
                              infer_state.b_start_loc,
                              infer_state.b_seq_len,
                              infer_state.max_len_in_batch)
        destindex_copy_kv(prefill_key_buffer, prefill_mem_index, mem_manager.key_buffer[self.layer_num_])
        destindex_copy_kv(prefill_value_buffer, prefill_mem_index, mem_manager.value_buffer[self.layer_num_])
        q = None
        o_tensor1 = torch.addmm(layer_weight.att_out_dense_bias_, o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_, beta=1.0 / self.world_size_, alpha=1.0)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(total_token_num, self.embed_dim_))
        o_tensor1 = None
        return


    def _token_flash_attention(self, input_embding, infer_state: StarcoderInferStateInfo, layer_weight: StarcoderTransformerLayerWeight):

        total_token_num = infer_state.total_token_num
        batch_size = infer_state.batch_size
        calcu_shape1 = (batch_size, self.tp_head_num_, self.head_dim_)
        if infer_state.decode_is_contiguous:
            cache_k = infer_state.mem_manager.key_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
            cache_v = infer_state.mem_manager.value_buffer[self.layer_num_][infer_state.decode_mem_start:infer_state.decode_mem_end, :, :]
        else:
            cache_k = infer_state.decode_key_buffer
            cache_v = infer_state.decode_value_buffer
            
        input1 = layernorm_forward(input_embding, weight=layer_weight.input_layernorm_weight_, bias=layer_weight.input_layernorm_bias_, eps=self.layer_norm_eps_)
        # import ipdb;ipdb.set_trace()
        q = torch.addmm(layer_weight.q_bias_, input1.view(-1, self.embed_dim_), layer_weight.q_weight_, beta=1.0, alpha=1.0)
        torch.addmm(layer_weight.k_bias_, input1.view(-1, self.embed_dim_), layer_weight.k_weight_, beta=1.0, alpha=1.0,
                    out=cache_k.view(-1, self.tp_kv_head_sum_dim_))
        torch.addmm(layer_weight.v_bias_, input1.view(-1, self.embed_dim_), layer_weight.v_weight_, beta=1.0, alpha=1.0,
                    out=cache_v.view(-1, self.tp_kv_head_sum_dim_))
        
        if not infer_state.decode_is_contiguous:
            destindex_copy_kv(cache_k, infer_state.decode_mem_index, infer_state.mem_manager.key_buffer[self.layer_num_])
            destindex_copy_kv(cache_v, infer_state.decode_mem_index, infer_state.mem_manager.value_buffer[self.layer_num_])
             
        input1 = None

        att_m_tensor = torch.empty((self.tp_head_num_, total_token_num), dtype=q.dtype, device="cuda")

        token_att_fwd(q.view(calcu_shape1),
                      infer_state.mem_manager.key_buffer[self.layer_num_],
                      att_m_tensor,
                      infer_state.b_loc,
                      infer_state.b_start_loc,
                      infer_state.b_seq_len,
                      infer_state.max_len_in_batch)

        prob = torch.empty_like(att_m_tensor)
        token_softmax_fwd(att_m_tensor, infer_state.b_start_loc, infer_state.b_seq_len, prob, infer_state.max_len_in_batch)
        att_m_tensor = None

        o_tensor = torch.empty_like(q)

        token_att_fwd2(prob,
                       infer_state.mem_manager.value_buffer[self.layer_num_],
                       o_tensor.view(calcu_shape1),
                       infer_state.b_loc,
                       infer_state.b_start_loc,
                       infer_state.b_seq_len,
                       infer_state.max_len_in_batch)
        prob = None
        q = None
        o_tensor1 = torch.addmm(layer_weight.att_out_dense_bias_, o_tensor.view(-1, self.tp_head_sum_dim_), layer_weight.att_out_dense_weight_, beta=1.0 / self.world_size_, alpha=1.0)
        o_tensor = None
        if self.world_size_ > 1:
            dist.all_reduce(o_tensor1, op=dist.ReduceOp.SUM, async_op=False)
        input_embding.add_(o_tensor1.view(batch_size, self.embed_dim_))
        o_tensor1 = None
        return