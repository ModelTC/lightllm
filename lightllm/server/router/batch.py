import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from lightllm.server.core.objs import ShmReqManager, Req
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class ReqDetokenizationState:
    def __init__(
        self,
        group_req_id: int,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
        skip_special_tokens: bool,
        add_spaces_between_special_tokens: bool,
        print_eos_token: bool,
        best_of: int,
    ) -> None:
        self.request_id = None
        self.group_req_id = group_req_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}
        self.skip_special_tokens = skip_special_tokens
        self.add_spaces_between_special_tokens = add_spaces_between_special_tokens
        self.print_eos_token = print_eos_token
        self.best_of = best_of


class Batch:
    def __init__(self, batch_id, reqs: List[Req], dp_size: int):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}
        self.dp_size = dp_size

        # 该参数只会在batch init， prefill， decode 后进行更新，并在剔除请求时减少
        # 在 batch rpc init 之后才会被填充正确的值，初始化为 None
        self.batch_decode_need_tokens = [None for _ in range(dp_size)]
        return

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def mark_and_get_finished_req_and_preupdate_status(self):
        unfinished_req_ids, finished_req_ids = [], []
        for req in self.reqs:
            if req.finish_status.is_finished():
                finished_req_ids.append(req.request_id)
                req_dp_index = req.sample_params.suggested_dp_index
                # 标记的时候，也同时更新一些这些请求被移除掉的更新量，有点dirty
                self.batch_decode_need_tokens[req_dp_index] -= req.get_decode_need_tokens()
            else:
                unfinished_req_ids.append(req.request_id)

        return unfinished_req_ids, finished_req_ids

    def filter_out_finished_req(self, unfinished_req_ids, finished_req_ids, shm_req_manager: ShmReqManager):
        # update batch
        if len(finished_req_ids) != 0:
            # 确保被回收, 减引用计数
            for req_id in finished_req_ids:
                req = self.id_to_reqs[req_id]
                shm_req_manager.put_back_req_obj(req)
                req = None

            self.reqs = [self.id_to_reqs[req_id] for req_id in unfinished_req_ids]
            self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def pop_req(self, req_id):
        self.reqs = [req for req in self.reqs if req.request_id != req_id]
        req = self.id_to_reqs[req_id]
        self.id_to_reqs.pop(req_id)
        req_dp_index = req.sample_params.suggested_dp_index
        self.batch_decode_need_tokens[req_dp_index] -= req.get_decode_need_tokens()
        return

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch: "Batch"):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        for dp_index in range(self.dp_size):
            self.batch_decode_need_tokens[dp_index] += mini_batch.batch_decode_need_tokens[dp_index]
        return

    def dp_merge(self, mini_batch: "Batch"):
        if mini_batch is None:
            return

        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def __repr__(self):
        return f"batch_id={self.batch_id}, " f"reqs={self.reqs}, "

    def simple_log(self):
        return f"batch_id={self.batch_id}, time:{time.time()}s req_ids:{[req.request_id for req in self.reqs]}"
