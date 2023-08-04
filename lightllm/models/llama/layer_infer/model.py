import os
import json
import torch
from .pre_layer_inference import PreLayerInfer
from .post_layer_inference import PostLayerInfer
from .transformer_layer_inference import TransformerLayerInfer
from ..layer_weights.pre_and_post_layer_weight import *
from ..layer_weights.transformer_layer_weight import *
from .infer_struct import InferStateInfo
from lightllm.models.llama.layer_weights.hf_load_utils import load_hf_weights
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.infer_utils import init_bloc
from lightllm.common import np_from_tensor
from typing import List, Dict

class LlamaTpPartModel:
    tp_rank_: int
    world_size_: int
    weight_dir_: str
    config: Dict
    mem_manager: MemoryManager
    pre_post_weight: PreAndPostLayerWeight
    trans_layers_weight: List[TransformerLayerWeight]
    pre_infer: PreLayerInfer
    post_infer: PostLayerInfer
    layers_infer: List[TransformerLayerInfer]
    head_num_: int
    head_dim_: int
    tp_head_num_: int
    vocab_size: int
    _cos_cached: torch.Tensor
    _sin_cached: torch.Tensor

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=""):
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.weight_dir_ = weight_dir
        with open(os.path.join(weight_dir, "config.json"), 'r') as json_file:
            self.config = json.load(json_file)

        assert load_way == "HF", "llama only support HF format to load Now!"
        assert mode == "", "future to support int8 int4 ..."
        
        self.mem_manager = MemoryManager(max_total_token_num, 
                                         dtype=torch.float16,
                                         head_num=self.config["num_attention_heads"] // world_size,
                                         head_dim=self.config["hidden_size"] // self.config["num_attention_heads"],
                                         layer_num=self.config["num_hidden_layers"])

        self.pre_post_weight = PreAndPostLayerWeight(self.tp_rank_, self.world_size_, torch.float16, self.config)
        self.trans_layers_weight = [
            TransformerLayerWeight(i, self.tp_rank_, self.world_size_, torch.float16, self.config, mode=mode)
            for i in range(self.config["num_hidden_layers"])
        ]

        load_hf_weights("fp16", weight_dir, pre_post_layer=self.pre_post_weight, transformer_layer_list=self.trans_layers_weight)

        self.pre_infer = PreLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.post_infer = PostLayerInfer(tp_rank=self.tp_rank_, world_size=self.world_size_, network_config=self.config)
        self.layers_infer = [
            TransformerLayerInfer(
                i,
                tp_rank=self.tp_rank_,
                world_size=self.world_size_,
                network_config=self.config,
                mode=mode
            ) for i in range(self.config["num_hidden_layers"])
        ]

        self.head_num_ = self.config["num_attention_heads"]
        self.head_dim_ = self.config["hidden_size"] // self.head_num_
        assert self.head_num_ % self.world_size_ == 0
        self.tp_head_num_ = self.head_num_ // self.world_size_
        self.vocab_size = self.config["vocab_size"]
        self.init_to_get_rotary()

    def init_to_get_rotary(self, base=10000):
        max_seq_len = self.config.get("max_position_embeddings", 2048)
        base = float(base)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim_, 2, device="cpu", dtype=torch.float32) / self.head_dim_))
        t = torch.arange(max_seq_len + 1024 * 64, device="cpu", dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        cos_sin_cached = torch.empty([2] + list(freqs.shape), dtype=torch.float16, device='cuda')
        cos_sin_cached[0][:] = torch.cos(freqs).to(torch.float16)
        cos_sin_cached[1][:] = torch.sin(freqs).to(torch.float16)
        self._cos_sin_cached = cos_sin_cached
        self._cos_cached = self._cos_sin_cached[0]
        self._sin_cached = self._cos_sin_cached[1]
        return

    @torch.no_grad()
    def forward(
            self,
            batch_size: int,
            total_token_num: int,
            max_len_in_batch: int,
            input_ids: torch.Tensor,
            b_loc: torch.Tensor,
            b_start_loc: torch.Tensor,
            b_seq_len: torch.Tensor,
            is_prefill=True):
        if is_prefill:
            assert (input_ids.shape[0] == total_token_num)
            assert (b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])
            infer_state = InferStateInfo()
            infer_state.is_prefill = is_prefill
            infer_state.batch_size = batch_size
            infer_state.total_token_num = total_token_num
            infer_state.max_len_in_batch = max_len_in_batch
            infer_state.b_loc = b_loc
            infer_state.b_start_loc = b_start_loc
            infer_state.b_seq_len = b_seq_len
            infer_state.mem_manager = self.mem_manager
            infer_state.prefill_mem_index = self.mem_manager.alloc(infer_state.total_token_num)

            b_seq_len_cpu = b_seq_len.cpu()

            _arange = np.arange(0, max_len_in_batch)
            b_seq_len_numpy = np_from_tensor(b_seq_len_cpu)
            position_ids = torch.from_numpy(np.concatenate([_arange[:x] for x in b_seq_len_numpy], axis=0)).cuda()
            del _arange

            position_cos_sin = torch.index_select(self._cos_sin_cached, 1, position_ids)
            infer_state.position_cos = position_cos_sin[0].view(position_ids.shape[0], -1)
            infer_state.position_sin = position_cos_sin[1].view(position_ids.shape[0], -1)
            del position_ids

            infer_state.prefill_key_buffer = torch.empty((infer_state.total_token_num, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            infer_state.prefill_value_buffer = torch.empty((infer_state.total_token_num, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
            init_bloc(b_loc, b_seq_len_numpy, max_len_in_batch, infer_state.prefill_mem_index)
            
            predict_logics = self._context_forward(input_ids, infer_state)
            return predict_logics
        else:
            infer_state = InferStateInfo()
            infer_state.is_prefill = is_prefill
            infer_state.batch_size = batch_size
            infer_state.total_token_num = total_token_num
            infer_state.max_len_in_batch = max_len_in_batch
            assert (batch_size == b_loc.shape[0] == b_start_loc.shape[0] == b_seq_len.shape[0])

            _sel = b_seq_len - 1
            position_cos_sin = torch.index_select(self._cos_sin_cached, 1, _sel)
            infer_state.position_cos = position_cos_sin[0].view(batch_size, -1)
            infer_state.position_sin = position_cos_sin[1].view(batch_size, -1)
            del _sel
            infer_state.b_loc = b_loc
            infer_state.b_start_loc = b_start_loc
            infer_state.b_seq_len = b_seq_len
            
            infer_state.mem_manager = self.mem_manager
            alloc_mem = self.mem_manager.alloc_contiguous(batch_size)
            if alloc_mem is not None:
                infer_state.decode_is_contiguous = True
                infer_state.decode_mem_index = alloc_mem[0]
                infer_state.decode_mem_start = alloc_mem[1]
                infer_state.decode_mem_end = alloc_mem[2]
                b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
            else:
                infer_state.decode_is_contiguous = False
                alloc_mem = self.mem_manager.alloc(batch_size)
                infer_state.decode_mem_index = alloc_mem
                infer_state.decode_key_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                infer_state.decode_value_buffer = torch.empty((batch_size, self.tp_head_num_, self.head_dim_), dtype=torch.float16, device="cuda")
                b_loc[:, max_len_in_batch - 1] = infer_state.decode_mem_index
            
            predict_logics = self._token_forward(input_ids, infer_state)
            return predict_logics
    
    def _context_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.context_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.config["num_hidden_layers"]):
            input_embs = self.layers_infer[i].context_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics

    def _token_forward(self, input_ids, infer_state: InferStateInfo):
        cuda_input_ids = input_ids
        input_embs = self.pre_infer.token_forward(cuda_input_ids, infer_state, self.pre_post_weight)
        for i in range(self.config["num_hidden_layers"]):
            input_embs = self.layers_infer[i].token_forward(input_embs, infer_state, self.trans_layers_weight[i])
        predict_logics = self.post_infer.token_forward(input_embs, infer_state, self.pre_post_weight, return_logics=True)
        return predict_logics
