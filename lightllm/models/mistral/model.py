import os
import json
import torch

from lightllm.common.basemodel import TpPartBaseModel
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer
from lightllm.models.mistral.infer_struct import MistralInferStateInfo
from lightllm.models.mistral.layer_infer.transformer_layer_infer import MistralTransformerLayerInfer
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

from lightllm.common.mem_utils import MemoryManager
from lightllm.common.infer_utils import init_req_to_token_indexes
from lightllm.common.basemodel.triton_kernel.copy_kv_index_to_req import copy_kv_index_to_req

class MistralTpPartModel(TpPartBaseModel):
    # weight class
    pre_and_post_weight_class = LlamaPreAndPostLayerWeight
    transformer_weight_class = LlamaTransformerLayerWeight

    # infer class
    pre_layer_infer_class = LlamaPreLayerInfer
    post_layer_infer_class = LlamaPostLayerInfer
    transformer_layer_infer_class = MistralTransformerLayerInfer

    # infer state class
    infer_state_class = MistralInferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_config(self):
        super()._init_config()
        # rename key [SYM: to be confirmed]
        return

    def _verify_params(self):
        assert self.load_way in ["HF"], "mistral only supports HF format to load Now!"
        assert self.config["num_key_value_heads"] % self.world_size_ == 0
        assert self.config["num_attention_heads"] % self.world_size_ == 0
        return

    def _init_custom(self):
        self._init_to_get_rotary()
        return

    def _init_mem_manager(self):
        self.mem_manager = MemoryManager(self.max_total_token_num, # [SYM] should be sliding window?
                                        dtype=torch.float16,
                                        head_num=self.config["num_key_value_heads"] // self.world_size_,
                                        head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                        layer_num=self.config["num_hidden_layers"],
                                        always_copy=False)       
        return

    def _init_to_get_rotary(self, default_base=10000):
        if self.config.get("rope_scaling", {}) is None:
            rope_scaling_factor = 1.0
        else:
            rope_scaling_factor = self.config.get("rope_scaling", {}).get("factor", 1.0)

        base = self.config.get("rope_theta", float(default_base))

        if "max_sequence_length" in self.config:
            max_seq_len = self.config["max_sequence_length"]
        else:
            max_position_embeddings = self.config.get(
                "max_position_embeddings",
                2048 if base <= 10000.0 + 1e-5 else 16384
            )
            max_seq_len = max_position_embeddings * rope_scaling_factor

        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32) / rope_scaling_factor
        freqs = torch.outer(t, inv_freq)

        self._cos_cached = torch.cos(freqs).to(torch.float16).cuda()
        self._sin_cached = torch.sin(freqs).to(torch.float16).cuda()
        return
    
    def _prefill(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = True
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.sliding_window = self.config["sliding_window"]
        assert (input_ids.shape[0] == total_token_num)
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len

        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(infer_state.total_token_num)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]

        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(infer_state.total_token_num)
            infer_state.mem_index = alloc_mem
            infer_state.key_buffer = torch.empty((infer_state.total_token_num, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.value_buffer = torch.empty((infer_state.total_token_num, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
        
        init_req_to_token_indexes(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len,
                            max_len_in_batch, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._context_forward(input_ids, infer_state)
        return predict_logics

    def _decode(self, batch_size, total_token_num, max_len_in_batch, input_ids, b_req_idx, b_start_loc, b_seq_len):
        infer_state = self.infer_state_class()
        infer_state.is_prefill = False
        infer_state.batch_size = batch_size
        infer_state.total_token_num = total_token_num
        infer_state.max_len_in_batch = max_len_in_batch
        infer_state.sliding_window = self.config["sliding_window"]
        assert (b_req_idx.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
        infer_state.b_req_idx = b_req_idx
        infer_state.b_start_loc = b_start_loc
        infer_state.b_seq_len = b_seq_len
        # [SYM] still reserve all kv cache
        infer_state.b_att_seq_len = b_seq_len.clone()
        infer_state.b_att_start_loc = b_start_loc.clone()
        infer_state.b_start_loc_window = b_start_loc.clone()
        infer_state.total_cache_num = 0
        for i in range(0, batch_size):
            if infer_state.sliding_window < infer_state.b_seq_len[i]:
                infer_state.b_start_loc_window[i] = infer_state.b_seq_len[i] - infer_state.sliding_window
                infer_state.b_att_seq_len[i] = infer_state.sliding_window
            else:
                infer_state.b_start_loc_window[i] = 0
                infer_state.b_att_seq_len[i] = infer_state.b_seq_len[i]
            infer_state.b_att_start_loc[i] = infer_state.total_cache_num
            infer_state.total_cache_num += infer_state.b_att_seq_len[i]
        
        infer_state.mem_manager = self.mem_manager
        infer_state.req_manager = self.req_manager

        alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
        if alloc_mem is not None:
            infer_state.mem_is_contiguous = True
            infer_state.mem_index = alloc_mem[0]
            infer_state.mem_start = alloc_mem[1]
            infer_state.mem_end = alloc_mem[2]
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)
        else:
            infer_state.mem_is_contiguous = False
            alloc_mem = self.mem_manager.alloc(batch_size)
            infer_state.mem_index = alloc_mem
            infer_state.key_buffer = torch.empty((batch_size, self.tp_k_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.value_buffer = torch.empty((batch_size, self.tp_v_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            copy_kv_index_to_req(self.req_manager.req_to_token_indexs, b_req_idx, b_seq_len, infer_state.mem_index)

        infer_state.init_some_extra_state(self, input_ids)
        predict_logics = self._token_forward(input_ids, infer_state)
        return predict_logics
    
    