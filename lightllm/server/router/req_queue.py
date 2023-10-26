import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from lightllm.utils.infer_utils import  calculate_time


class ReqQueue:

    def __init__(self, max_total_tokens, allow_finish_percent, batch_max_tokens, running_max_req_size, max_new_token_len, token_ratio) -> None:
        self.max_total_tokens = max_total_tokens
        self.allow_finish_percent = allow_finish_percent
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.token_ratio = token_ratio
        self.max_new_token_len = max_new_token_len
        self.pending_count = 0

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def _init_cache_list(self, current_batch:Batch):
        if current_batch is not None:
            self.cache_len_list = [(req.input_len + len(req.output_ids) - 1, max(1, self.max_new_token_len - len(req.output_ids))) for req in current_batch.reqs]
        else:
            self.cache_len_list = []

    def _can_add_new_req(self, req, token_ratio=0.):
        self.cache_len_list.append((req.calcu_need_tokens(), max(1, self.max_new_token_len - len(req.output_ids) - 1))) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        percent = self.allow_finish_percent if token_ratio <= self.token_ratio else 1
        ensure_num = min(int(len(self.cache_len_list) * (1 - percent)), len(self.cache_len_list) - 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array)[ensure_num:].max()
        if need_max_token_num < self.max_total_tokens and len(self.cache_len_list) <= self.running_max_req_size:
            return True
        else:
            return False

    def generate_new_batch(self, current_batch:Batch, token_ratio=0.):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        restore = True if self.pending_count > 0 else False
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if restore and self.pending_count == 0:
                break
            if self._can_add_new_req(req, token_ratio) and new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                if restore:
                    self.pending_count -= 1
                    req.offload = False
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return restore, new_batch
        else:
            return restore, None

    def calcu_stopd_tokens(self):
        stopd_tokens = 0
        for i in range(self.pending_count):
            stopd_tokens += self.waiting_req_list[i].calcu_used_tokens()
        return stopd_tokens

    def insert(self, req):
        self.waiting_req_list.insert(self.pending_count, req)
        self.pending_count += 1
    
    def is_waiting_list_empty(self):
        return len(self.waiting_req_list) == 0

    def has_pending_reqs(self):
        if self.pending_count > 0:
            return True
        return False