from typing import List
from lightllm.server.core.objs import Req


class DecodeReq:
    def __init__(
        self,
        req: Req,
    ) -> None:
        self.request_id = req.request_id
        self.group_req_id = req.group_req_id
        self.prompt_ids = req.shm_prompt_ids.arr[0 : req.input_len].tolist()
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.req = req
        self.input_len = self.req.input_len
        self.prefix_str = ""

    def need_detoken(self):
        if (not self.req.is_aborted) and len(self.output_ids) < self.req.candetoken_out_len:
            return True
        return False

    def out_queue_is_full(self):
        return self.req.out_tokens_queue.is_full()

    def get_next_token_id_and_index(self):
        src_index = self.input_len + len(self.output_ids)
        return self.req.shm_prompt_ids.arr[src_index], src_index

    def can_set_release_mark(self):
        if self.req.is_aborted:
            return True
        if (
            self.req.finish_status.is_finished()
            and self.req.candetoken_out_len == len(self.output_ids)
            and self.req.finish_token_index == self.input_len + len(self.output_ids) - 1
        ):
            return True
        return False
