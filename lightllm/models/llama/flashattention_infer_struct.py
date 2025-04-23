import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index


class FlashAttentionStateInfo(LlamaInferStateInfo):
    _shared_page_table_buffer = None

    def __init__(self):
        super().__init__()

    @classmethod
    def set_page_table_buffer(cls, graph_max_batch_size: int, max_seq_len: int):
        cls._shared_page_table_buffer = [
            torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
            torch.empty(graph_max_batch_size * max_seq_len, dtype=torch.int32).to(get_current_device_id()),
        ]

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        if self.is_prefill:
            # TODO: Prefill Attention
            pass
        else:
            if self._shared_page_table_buffer is None:
                FlashAttentionStateInfo.set_page_table_buffer(
                    graph_max_batch_size=model.graph_max_batch_size,
                    max_seq_len=model.graph_max_len_in_batch,
                )
            # Meta information of flashattention for decoding
            self.cu_seqlens_q = torch.arange(0, self.batch_size + 1, dtype=torch.int32, device=input_ids.device)
            self.cu_seqlens_k = torch.nn.functional.pad(self.b_start_loc, (1, 0))
            if self.batch_size <= model.graph_max_batch_size and self.max_len_in_batch <= model.graph_max_len_in_batch:
                self.page_table = self._shared_page_table_buffer[self.microbatch_index][
                    : self.batch_size * self.max_len_in_batch
                ].reshape(self.batch_size, self.max_len_in_batch)
            else:
                self.page_table = torch.empty((self.batch_size, self.max_len_in_batch), dtype=torch.int32).to(
                    input_ids.device
                )

            self.page_table.copy_(model.req_manager.req_to_token_indexs[self.b_req_idx, : self.max_len_in_batch])
        return
