import uuid
import numpy as np
from typing import List, Tuple
from lightllm.server.io_struct import Batch, Req

class WaitStrategyRegistry:
    strategy = {}
    
    @classmethod
    def getStrategy(cls, name, max_total_token_num, ema, *args, **kwargs):
        return cls.strategy[name](max_total_token_num, ema, args, kwargs)

class Meta(type):
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        WaitStrategyRegistry.strategy[name] = cls
        return cls

class WaitStrategy:

    def __init__(self, max_total_token_num, ema, *args, **kwargs) -> None:
        self.max_total_token_num = max_total_token_num
        self.ema = ema
        self.moving_max_new_tokens = False
        self.stoped_req_list: List[Req] = []

    def append_to_stoped(self, req):
        self.stoped_req_list.append(req)
        return

    def _order(self, req: Req):
        raise NotImplementedError

    def _ordering_index(self, batch: Batch):
        req_index = range(len(batch.reqs))
        return sorted(req_index, key=lambda i: self._order(batch.reqs[i]), reverse=True)
    
    def _selection(self, batch: Batch):
        raise NotImplementedError

    def restore_batch(self, batch: Batch):
        raise NotImplementedError
    
    def select_reqs(self, batch: Batch):
        request_ids = self._selection(batch)
        req_id_list = []
        for request_id in request_ids:
            req = batch.pop(request_id)
            assert req is not None
            self.append_to_stoped(req)
            req_id_list.append(request_id)
        if len(batch.reqs)  == 0:
            raise RuntimeError(f"No enough memory to run, current batch size {len(req_id_list)}")
        return req_id_list
    
    def _calcu_max_tokens(self, req_len_list: List[Tuple[int, int]]):
        if not req_len_list:
            return -1
        left_len = np.array([e[1] for e in req_len_list])
        run_len = np.array([e[0] for e in req_len_list])
        cum_run_len = np.cumsum(run_len)
        size_array = np.arange(1, len(req_len_list) + 1, 1)
        return (left_len * size_array + cum_run_len).max()

    def can_decode(self, batch: Batch):
        raise NotImplementedError

    def is_stoped_list_empty(self):
        return len(self.stoped_req_list) == 0

    def calcu_stopd_prompt_tokens(self):
        stopd_prompt_tokens = 0
        for req in self.stoped_req_list:
            stopd_prompt_tokens += req.input_len
        return stopd_prompt_tokens

    def calcu_stopd_output_tokens(self):
        stopd_output_tokens = 0
        for req in self.stoped_req_list:
            stopd_output_tokens += len(req.output_ids) - 1
        return stopd_output_tokens

    def calcu_stopd_tokens(self):
        raise NotImplementedError


class SJF(WaitStrategy, metaclass=Meta):
    """ Shortest job first
    """
    
    def __init__(self, max_total_token_num, *args, **kwargs) -> None:
        super().__init__(max_total_token_num, *args, **kwargs)
        self.token_raio = 0.8
        if "token_ratio" in kwargs:
            self.token_raio = kwargs["token_ratio"]

    def _order(self, req: Req):
        return req.max_output_len
    
    def calcu_stopd_tokens(self):
        stopd_tokens = 0
        for req in self.stoped_req_list:
            stopd_tokens += req.calcu_used_tokens()
        return stopd_tokens

    def can_decode(self, batch: Batch):
        remaining_tokens = self.max_total_token_num - batch.calcu_used_tokens() - self.calcu_stopd_tokens()
        return len(batch.reqs) <= remaining_tokens

    def restore_batch(self, batch: Batch):
        can_restore_list = []
        for req in self.stoped_req_list:
            can_restore_list.append(req)
        self.stoped_req_list = []
        if len(can_restore_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_restore_list)
            return new_batch
        else:
            return None

    def _selection(self, batch: Batch):
        remain_tokens = self.max_total_token_num - (batch.calcu_used_tokens() + self.calcu_stopd_tokens())
        sorted_index = self._ordering_index(batch)
        req_len_list = [(batch.reqs[index].input_len + len(batch.reqs[index].output_ids) - 1, 
                     self.ema.get_max_output_len(batch.reqs[index]) - len(batch.reqs[index].output_ids)) for index in sorted_index]
        for i in range(len(req_len_list)):
            top_req = req_len_list.pop(0)
            update_token_num = remain_tokens - top_req[0]
            if self._calcu_max_tokens(req_len_list) <= update_token_num:
                break
        select_index = sorted_index[:(i+1)]
        return [batch.reqs[sorted_index[j]].request_id for j in select_index]
