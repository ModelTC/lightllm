import bisect
from collections import deque
import random
import uuid
import asyncio
import numpy as np
from typing import List, Tuple
from ..io_struct import Batch, Req
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Req
from lightllm.server.io_struct import ReqRunStatus, FinishStatus

class ReqQueue:

    def __init__(self, args, prompt_cache_used_tokens, prompt_cache_req_num) -> None:
        self.max_total_tokens = args.max_total_token_num
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.router_token_ratio = args.router_token_ratio
        self.router_max_new_token_len = args.router_max_new_token_len
        self.pause_req_dict = {} # 用于保存队列中被暂停的请求，暂停原因为 ReqRunStatus.PAUSED_AND_KVKEEP  ReqRunStatus.PAUSED_AND_OFFLOAD
        self.pause_req_used_tokens = 0

        self.is_splitfuse_mode = args.splitfuse_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        # 当使用 prompt cache 特性时的维护变量
        self.prompt_cache_used_tokens = prompt_cache_used_tokens
        self.prompt_cache_req_num = prompt_cache_req_num
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def back_to_wait_list(self, req_list:List[Req]):
        for req in req_list:
            if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                self.pause_req_dict[req.request_id] = req
        self.waiting_req_list = req_list + self.waiting_req_list
        self.recalcu_pause_req_used_tokens()
        return 

    def _init_cache_list(self, current_batch:Batch, is_busy):
        self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        self.cache_pause_reqs_num = len(self.pause_req_dict) 
        if current_batch is not None:
            self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs]
        else:
            self.cache_len_list = []

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req:Req, is_busy):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens -= req.get_used_tokens()
            self.cache_pause_reqs_num -= 1

        ok_token_num = need_max_token_num < self.max_total_tokens - self.cache_pause_reqs_used_tokens - self.prompt_cache_used_tokens
        ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num + self.prompt_cache_req_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False
    
    #@calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch:Batch):

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.prompt_cache_req_num
        exist_req_num += 0 if current_batch is None else len(current_batch.reqs)
        exist_req_num += len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None
        
        # 计算当前所有的token使用量，包括当前使用和暂停的
        cur_all_used_tokens = 0 if current_batch is None else current_batch.batch_used_tokens
        cur_all_used_tokens += self.recalcu_pause_req_used_tokens() + self.prompt_cache_used_tokens
        
        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio

        # 得到当前batch 往前 decode 一次，需要的token量，在 splitfuse 模式下才有用，因为splitfuse
        # 模式下 类似prefill 和 deocde 是在一起进行的，所以需要合并考虑。
        # 普通模式是 先prefill 后 decode，所以只考虑prefill的时候 token使用量不要超过限制。
        if not self.is_splitfuse_mode:
            cur_batch_decode_need_tokens = 0
        else:
            cur_batch_decode_need_tokens = 0 if current_batch is None else current_batch.batch_decode_need_tokens
        
        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0 # 主要是对 prefill 或者 splitfuse 大块计算时候的限制
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.finish_status.is_aborted() and req.req_status == ReqRunStatus.WAIT_IN_QUEUE: 
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉. 
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                continue
            req_first_router_need_tokens = req.get_first_router_need_tokens()
            if cur_batch_decode_need_tokens + new_batch_first_router_need_tokens + req_first_router_need_tokens <= self.batch_max_tokens and self._can_add_new_req(req, is_busy) :
                can_run_list.append(req)
                new_batch_first_router_need_tokens += req_first_router_need_tokens
                if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            # 生成新 batch 以后，更新一下状态
            self.recalcu_pause_req_used_tokens()
            return new_batch
        else:
            return None
        
    def recalcu_pause_req_used_tokens(self):
        used_tokens = 0
        for req_id, req_obj in self.pause_req_dict.items():
            used_tokens += req_obj.get_used_tokens()
        self.pause_req_used_tokens = used_tokens
        return self.pause_req_used_tokens


class FuturePastReqQueue(ReqQueue):
    WINDOW_SIZE = 200
    MINIMUM_SAMPLES = 200
    MAXIMUM_LISTS = 5
    REVERSED = 0.05

    def __init__(self, args, prompt_cache_used_tokens, prompt_cache_req_num) -> None:
        super().__init__(args, prompt_cache_used_tokens, prompt_cache_req_num)
        initial_len = args.max_req_total_len - args.max_req_input_len
        self.history_output_len = deque([initial_len] * (self.WINDOW_SIZE // 2), maxlen=self.WINDOW_SIZE)

    def _sample_cache_list(self, reqs: List[Req], samples=1) -> List[List[Tuple[int, int]]]:
        cache_len_lists = [[] for _ in range(samples)]
        his_Lo = sorted(self.history_output_len)
        for req in reqs:
            dl = len(req.output_ids)
            pos = bisect.bisect(his_Lo, dl)
            sample_range = [dl] + his_Lo[pos:]
            if sample_range[-1] < req.max_output_len:
                sample_range.append(req.max_output_len)

            for i in range(samples):
                random_p = np.random.random() * (len(sample_range)-1)
                l_pos = int(random_p)
                l_val, r_val = sample_range[l_pos:l_pos+2]

                # 线性差值
                sampled = round(l_val + (r_val - l_val) * (random_p - l_pos))
                cache_len_lists[i].append(req.get_tuple_tokens(False, sampled, minimal_output_len_factor=1))

        return cache_len_lists

    def _calc_max_token_num_needed(self, cache_len_list: List[Tuple[int, int]]) -> int:
        cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in cache_len_list])
        has_run_len_array = np.array([e[0] for e in cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num

    def _init_cache_list(self, current_batch:Batch, is_busy):
        self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        self.cache_pause_reqs_num = len(self.pause_req_dict)
        if current_batch is not None:
            n_lists = min(self.MAXIMUM_LISTS, int(self.MINIMUM_SAMPLES / len(current_batch.reqs)) + 1)
            self._cache_len_lists = self._sample_cache_list(current_batch.reqs, samples=n_lists)
        else:
            self._cache_len_lists = [[]]
        self.cache_len_list = self._cache_len_lists[0]   # keep compatibility

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req: Req, is_busy: bool):
        need_max_token_nums = []
        for li in self._cache_len_lists:
            newreq_output_len_sample = random.choice(self.history_output_len)
            li.append(req.get_tuple_tokens(False, newreq_output_len_sample, minimal_output_len_factor=1))
            need_max_token_nums.append(self._calc_max_token_num_needed(li))
        need_max_token_num = np.max(need_max_token_nums)

        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens -= req.get_used_tokens()
            self.cache_pause_reqs_num -= 1

        ok_token_num = need_max_token_num < self.max_total_tokens * (1 - self.REVERSED) - self.cache_pause_reqs_used_tokens - self.prompt_cache_used_tokens
        ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num + self.prompt_cache_req_num <= self.running_max_req_size

        return ok_token_num and ok_req_num

    def record_output_lengths(self, lengths: List[int]):
        self.history_output_len.extend(lengths)
