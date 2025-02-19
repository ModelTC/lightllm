import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.deepseek2.triton_kernel.repack_kv_index import repack_kv_index
import flashinfer


class Deepseek2InferStateInfo(LlamaInferStateInfo):
    def __init__(self):
        super().__init__()
        self.kv_starts = None
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        self.enable_flashinfer_decode_mla = os.getenv("ENABLE_FLASHINFER_DECODE_MLA", "False").upper() in [
            "ON",
            "TRUE",
            "1",
        ]

    def init_some_extra_state(self, model, input_ids: torch.Tensor):
        super().init_some_extra_state(model, input_ids)
        # 只有 decode 阶段使用 ppl 的优化算子才会有这个管理变量
        if not self.is_prefill:
            self.kv_starts = torch.cat([self.b_start_loc, self.b_start_loc[-1:] + self.b_seq_len[-1:]], dim=0)
            self.total_token_num_tensor = torch.sum(self.b_seq_len)
            if self.enable_flashinfer_decode_mla:
                self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(input_ids.device)
                self.q_indptr = torch.arange(self.batch_size + 1, dtype=torch.int32).to(input_ids.device)
                self.kv_indices = torch.empty(self.batch_size * model.max_seq_length, dtype=torch.int32).to(
                    input_ids.device
                )
                repack_kv_index(
                    self.req_manager.req_to_token_indexs,
                    self.b_req_idx,
                    self.b_seq_len,
                    self.b_start_loc,
                    self.max_len_in_batch,
                    self.kv_indices,
                )
                self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
                    self.workspace_buffer,
                    backend="fa2",
                    use_cuda_graph=True,
                    qo_indptr=self.q_indptr,
                    kv_indices=self.kv_indices,
                    kv_indptr=self.kv_starts,
                    kv_len_arr=self.b_seq_len,
                )
                self.head_num = model.tp_q_head_num_ * model.world_size_ if self.enable_dp else model.tp_q_head_num_
                self.kv_lora_rank = model.kv_lora_rank
                self.qk_rope_head_dim = model.qk_rope_head_dim
                self.softmax_scale = model.softmax_scale
                self.q_data_type = model.data_type
                self.kv_data_type = model.data_type
                self.wrapper.plan(
                    self.q_indptr,
                    self.kv_starts,
                    self.kv_indices,
                    self.b_seq_len,
                    self.head_num,
                    self.kv_lora_rank,
                    self.qk_rope_head_dim,
                    1,
                    False,  # causal
                    self.softmax_scale,
                    self.q_data_type,
                    self.kv_data_type,
                )

        if self.is_prefill:
            self.b_kv_start_loc = self.b_seq_len.cumsum(dim=0) - self.b_seq_len

        if self.enable_dp:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_token_num = input_ids.size(0)
            all_token_num = [torch.zeros(1, dtype=torch.int32).to(input_ids.device) for _ in range(world_size)]
            dist.all_gather(all_token_num, torch.tensor([local_token_num], dtype=torch.int32).to(input_ids.device))
            all_token_num = torch.cat(all_token_num, dim=0)
            self.all_token_num = all_token_num.sum().cpu().numpy()
            cumsum_token_num = torch.cumsum(all_token_num, dim=0).cpu().numpy()
            self.all_start_idx = cumsum_token_num - all_token_num.cpu().numpy()
            self.all_end_idx = cumsum_token_num
            self.start_idx = self.all_start_idx[rank]
            self.end_idx = self.all_end_idx[rank]

        return

    def copy_for_cuda_graph(self, new_infer_state):
        super().copy_for_cuda_graph(new_infer_state)
        if self.enable_flashinfer_decode_mla:
            self.wrapper.plan(
                self.q_indptr,
                self.kv_starts,
                self.kv_indices,
                self.b_seq_len,
                self.head_num,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                False,  # causal
                self.softmax_scale,
                self.q_data_type,
                self.kv_data_type,
            )
        return
