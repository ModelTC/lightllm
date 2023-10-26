import uuid
import numpy as np
from typing import List, Tuple
from lightllm.server.io_struct import Batch, Req

class Strategy:

    def _order(self, req: Req):
        raise NotImplementedError

    def ordering_reqs(self, batch: Batch):
        request_ids = [req.request_id for req in batch.reqs]
        return sorted(request_ids, key=lambda i: self._order(batch.id_to_reqs[i]), reverse=True)

class Fcfs(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def _order(self, req: Req):
        return req.rank

class Sfj(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def _order(self, req: Req):
        return req.max_output_len

class Hrnn(Strategy):

    def __init__(self) -> None:
        super().__init__()

    def _order(self, req: Req):
        return (req.input_len + req.max_output_len) / req.input_len

class SelectionManager:
    selections = {}

    @classmethod
    def getSelection(cls, strategy, req_queue, max_total_token_num, *args, **kwargs):
        if strategy == "hrnn":
            scheduler = Hrnn()
        elif strategy == "sfj":
            scheduler = Sfj()
        else:
            scheduler = Fcfs()
        return cls.selections["Selection"](scheduler, req_queue, max_total_token_num, args, kwargs)

class Meta(type):
    def __new__(meta, name, bases, attrs):
        cls = type.__new__(meta, name, bases, attrs)
        SelectionManager.selections[name] = cls
        return cls

class Selection(metaclass=Meta):

    def __init__(self, strategy, req_queue, max_total_token_num, *args, **kwargs) -> None:
        self.strategy = strategy
        self.req_queue = req_queue
        self.max_total_token_num = max_total_token_num
        self.offload = kwargs.get("offload", False)

    def select_reqs(self, batch: Batch):
        request_ids = self.strategy.ordering_reqs(batch)
        count = 0
        while True:
            remain_tokens = self.max_total_token_num - (batch.calcu_used_tokens() + self.req_queue.calcu_stopd_tokens())
            if len(batch.reqs) <= remain_tokens or len(request_ids) - 1 == count:
                break
            request_id = request_ids[count]
            req = batch.pop(request_id) if not self.offload else batch.offload(request_id)
            assert req is not None
            self.req_queue.insert(req)
            count += 1
        return request_ids[:count]

