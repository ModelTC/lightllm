import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.spec_info import SpeculativeDecodeAlgorithm

class Deepseek2InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.kv_starts = None

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        if not self.is_prefill:
            self.kv_starts = self.b1_cu_kv_seq_len

        if self.is_prefill:
            self.b1_kv_start_loc = self.b1_cu_kv_seq_len
            self.max_value_in_b_seq_len = self.b_seq_len.max().item()
        
        if not self.is_prefill and not self.spec_algo.is_none():
            b_seq_len_numpy = self.b_seq_len.cpu().numpy()
            position_ids = torch.from_numpy(
                np.concatenate(
                    [np.arange(b_seq_len_numpy[i] - 2, b_seq_len_numpy[i]) for i in range(len(b_seq_len_numpy))],
                    axis=0,
                )
            ).cuda()
            self.position_cos = torch.index_select(model._cos_cached, 0, position_ids).view(position_ids.shape[0], -1)
            self.position_sin = torch.index_select(model._sin_cached, 0, position_ids).view(position_ids.shape[0], -1)
        return