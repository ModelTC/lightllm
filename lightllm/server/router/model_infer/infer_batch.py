import torch
import numpy as np
import collections

from lightllm.common.configs.config import setting
from dataclasses import dataclass, field
from typing import List, Dict
from lightllm.common.req_manager import ReqManager
from lightllm.utils.infer_utils import mark_start, mark_end
import time


import time
from functools import wraps
 
# 装饰器函数
def print_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration_time = end_time - start_time
        print("execute time running %s: %s seconds" % (func.__name__, duration_time))
        return result
 
    return wrapper

class InferSamplingParams:

    def __init__(
        self,
        do_sample: bool = False,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        vocab_size: int = -1,
    ) -> None:
        self.do_sample = do_sample
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if self.top_k == -1:
            self.top_k = vocab_size
        return


class InferReq:

    def __init__(
        self,
        input_id,
        out_token_id_count={},
        sampling_param=None,
        b_req_idx=-1,
        b_seq_len=0,
    ) -> None:
        self.input_id = input_id
        self.out_token_id_count = out_token_id_count
        self.sampling_param = sampling_param
        self.b_req_idx = b_req_idx
        self.b_seq_len = b_seq_len
        return


@dataclass
class InferBatch:
    batch_id: int
    request_ids: List
    requests_mapping: Dict[int, int]
    input_ids: torch.Tensor

    nopad_total_token_num: int
    nopad_max_len_in_batch: int
    nopad_b_req_idx: torch.Tensor
    nopad_b_start_loc: torch.Tensor
    nopad_b_seq_len: torch.Tensor
    req_manager: ReqManager
    
    @classmethod
    @torch.no_grad()
    def init_batch(cls, batch_id, requests, dtype: torch.dtype, device: torch.device, req_manager:ReqManager, vocab_size: int):

        request_ids = []
        all_input_ids = []
        requests_mapping = {}
        
        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        b_start_loc = 0
        nopad_b_req_idx = req_manager.alloc(len(requests))
        nopad_b_start_loc = torch.zeros(len(requests), dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.zeros(len(requests), dtype=torch.int32, device='cuda')

        for i, r in enumerate(requests):
            # request id -> idx in list mapping

            tokenized_input = r['input_id']
            input_length = len(tokenized_input)
            all_input_ids.append(tokenized_input)
            request_ids.append(r['request_id'])
            # postprocessor
            sampling_param = r["sampling_param"]
            sampling_param["vocab_size"] = vocab_size
            nopad_total_token_num += input_length
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_length)  

            requests_mapping[r['request_id']] = InferReq(input_id=tokenized_input, out_token_id_count=collections.defaultdict(int), sampling_param=InferSamplingParams(**sampling_param), b_req_idx=nopad_b_req_idx[i], b_seq_len=input_length)
            
            nopad_b_seq_len[i] = input_length
            nopad_b_start_loc[i] = b_start_loc
            b_start_loc += input_length

        if len(requests) > 1:
            input_ids = np.concatenate(all_input_ids, dtype=np.int64)
        else:
            input_ids = all_input_ids[0]

        # Create tensors on device
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=device)

        return cls(
            batch_id=batch_id,
            request_ids=request_ids,
            requests_mapping=requests_mapping,
            input_ids=input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_req_idx=nopad_b_req_idx,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            req_manager=req_manager,
        )
    
    @print_info
    @torch.no_grad()
    def free_self(self):
        self.req_manager.free(self.nopad_b_req_idx, self.nopad_b_seq_len - 1)
        return
    
    @print_info
    @torch.no_grad()
    def filter(self, request_ids: List[str], finished_request_ids: List[str]):
        if len(self.requests_mapping) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self) or len(request_ids) == 0:
            return self
        
        requests_mapping = {}
        idx = 0
        b_start_loc = 0
        nopad_total_token_num = 0
        nopad_max_len_in_batch = 0
        input_ids = torch.zeros(len(request_ids), dtype=torch.int64, device='cuda')
        nopad_b_start_loc = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        nopad_b_req_idx = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        nopad_b_seq_len = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        for request_id in request_ids:
            req = self.requests_mapping[request_id]
            input_ids[idx] =req.input_id
            nopad_b_seq_len[idx] = req.b_seq_len
            nopad_b_req_idx[idx] = req.b_req_idx
            nopad_total_token_num += req.b_seq_len
            nopad_max_len_in_batch = max(req.b_seq_len, nopad_max_len_in_batch)
            nopad_b_start_loc[idx] = b_start_loc
            requests_mapping[request_id] = req
            idx +=1
            b_start_loc += req.b_seq_len
        
        for request_id in finished_request_ids:
            req = self.requests_mapping[request_id]
            self.req_manager.free_req(req.b_req_idx, req.b_seq_len - 1)

        return InferBatch(
            batch_id=self.batch_id,
            request_ids=request_ids,
            requests_mapping=requests_mapping,
            input_ids=input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_req_idx=nopad_b_req_idx,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            req_manager=self.req_manager,
        )

    @torch.no_grad()
    def stop_reqs(self, stop_request_ids: List[str]):
        stop_request_ids_set = set(stop_request_ids)
        request_ids = [i for i in self.requests_mapping.keys() if i not in stop_request_ids_set]
        b_start_loc = 0
        self.nopad_total_token_num = 0
        self.nopad_max_len_in_batch = 0
        self.input_ids = torch.zeros(len(request_ids), dtype=torch.int64, device='cuda')
        self.nopad_b_start_loc = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        self.nopad_b_req_idx = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        self.nopad_b_seq_len = torch.zeros(len(request_ids), dtype=torch.int32, device='cuda')
        self.request_ids = request_ids
        for i, request_id in enumerate(request_ids):
            req = self.requests_mapping[request_id]
            self.input_ids[i] =req.input_id
            self.nopad_b_seq_len[i] = req.b_seq_len
            self.nopad_b_req_idx[i] = req.b_req_idx
            self.nopad_total_token_num += req.b_seq_len
            self.nopad_max_len_in_batch = max(req.b_seq_len, self.nopad_max_len_in_batch)
            self.nopad_b_start_loc[i] = b_start_loc
            b_start_loc += req.b_seq_len
        return self

    @torch.no_grad()
    def restore_reqs(self, restore_request_ids: List[str]):
        restore_batch = InferBatch.merge(self, self.stop_son_batch)
        del self.stop_son_batch
        self.stop_son_batch = None
        self.input_ids = torch.zeros(len(self) + len(restore_request_ids), dtype=torch.int64, device='cuda')
        self.nopad_b_start_loc = torch.zeros(len(self) + len(restore_request_ids), dtype=torch.int32, device='cuda')
        self.nopad_b_req_idx = torch.zeros(len(self) + len(restore_request_ids), dtype=torch.int32, device='cuda')
        self.nopad_b_seq_len = torch.zeros(len(self) + len(restore_request_ids), dtype=torch.int32, device='cuda')
        b_start_loc = self.nopad_total_token_num
        cur_bs = len(self)
        for i, request_id in enumerate(restore_request_ids):
            req = self.requests_mapping[request_id]
            self.input_ids[cur_bs] =req.input_id
            self.nopad_b_seq_len[cur_bs] = req.b_seq_len
            self.nopad_b_req_idx[cur_bs] = req.b_req_idx
            self.nopad_total_token_num += req.b_seq_len
            self.nopad_max_len_in_batch = max(req.b_seq_len, self.nopad_max_len_in_batch)
            self.nopad_b_start_loc[cur_bs] = b_start_loc
            b_start_loc += req.b_seq_len
        self.request_ids += restore_request_ids
        return self

    @classmethod
    @torch.no_grad()
    def merge(cls, batch1, batch2):
        start_time = time.time()
        request_ids = batch1.request_ids + batch2.request_ids
        batch1.requests_mapping.update(batch2.requests_mapping)
        new_batch_size = len(batch1) + len(batch2)

        cumulative_batch_size = 0
        nopad_total_token_num = batch1.nopad_total_token_num + batch2.nopad_total_token_num
        nopad_max_len_in_batch = max(batch1.nopad_max_len_in_batch, batch2.nopad_max_len_in_batch)
        
        nopad_b_req_idx = torch.cat([batch1.nopad_b_req_idx, batch2.nopad_b_req_idx], dim=0)
        nopad_b_seq_len = torch.cat([batch1.nopad_b_seq_len, batch2.nopad_b_seq_len], dim=0)
        input_ids = torch.cat([batch1.input_ids, batch2.input_ids], dim=0)
        nopad_b_start_loc = torch.cat([batch1.nopad_b_start_loc, batch2.nopad_b_start_loc + batch1.nopad_total_token_num], dim=0)
        end_time = time.time()
        duration_time = end_time - start_time
        print("execute time running %s: %s seconds" % ("merge", duration_time))

        return InferBatch(
            batch_id=batch1.batch_id,
            request_ids=request_ids,
            requests_mapping=batch1.requests_mapping,
            input_ids=input_ids,
            nopad_total_token_num=nopad_total_token_num,
            nopad_max_len_in_batch=nopad_max_len_in_batch,
            nopad_b_req_idx=nopad_b_req_idx,
            nopad_b_start_loc=nopad_b_start_loc,
            nopad_b_seq_len=nopad_b_seq_len,
            req_manager=batch1.req_manager,
        )

    def __len__(self):
        return len(self.request_ids)
    
    
    def get_post_sample_tensors(self):
        presence_penalties: List[float] = []
        frequency_penalties: List[float] = []
        temperatures: List[float] = []
        top_ps: List[float] = []
        top_ks: List[int] = []
        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_seq_len: List[int] = [0,]
        p_max_len_in_batch: int = 0
        for i, request_id in enumerate(self.request_ids):
            id_to_count = self.requests_mapping[request_id].out_token_id_count
            sample_param = self.requests_mapping[request_id].sampling_param
            presence_penalties.append(sample_param.presence_penalty)
            frequency_penalties.append(sample_param.frequency_penalty)
            temperatures.append(sample_param.temperature)
            top_ps.append(sample_param.top_p)
            top_ks.append(sample_param.top_k)
            
            for token_id, count in id_to_count.items():
                p_token_ids.append(token_id)
                p_token_counts.append(count)
            p_seq_len.append(len(id_to_count))
            p_max_len_in_batch = max(p_max_len_in_batch, len(id_to_count))
        
        presence_penalties = torch.tensor(presence_penalties, dtype=torch.float, device="cuda")
        frequency_penalties = torch.tensor(frequency_penalties, dtype=torch.float, device="cuda")
        temperatures = torch.tensor(temperatures, dtype=torch.float, device="cuda")
        top_ps = torch.tensor(top_ps, dtype=torch.float, device="cuda")
        top_ks = torch.tensor(top_ks, dtype=torch.int32, device="cuda")
        p_token_ids = torch.tensor(p_token_ids, dtype=torch.int32, device="cuda")
        p_token_counts = torch.tensor(p_token_counts, dtype=torch.int32, device="cuda")
        p_seq_len = torch.tensor(p_seq_len, dtype=torch.int32, device="cuda")
        p_cumsum_seq_len = torch.cumsum(p_seq_len, dim=0, dtype=torch.int32)
        return presence_penalties, frequency_penalties, temperatures, top_ps, top_ks, p_token_ids, p_token_counts, p_cumsum_seq_len, p_max_len_in_batch