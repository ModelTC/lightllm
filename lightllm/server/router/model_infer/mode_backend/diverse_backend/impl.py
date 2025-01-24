import torch
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import (
    g_infer_context,
    InferReq,
    InferReqGroup,
    InferSamplingParams,
)
from typing import List, Tuple
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.tokenizer import get_tokenizer
from .pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from .post_process import sample


class DiversehBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        pass

    def build_group(self, req_ids: List[int]):
        for r_id in req_ids:
            req: InferReq = g_infer_context.requests_mapping[r_id]
            group_req_id = req.shm_req.group_req_id
            best_of = req.shm_req.sample_params.best_of
            if group_req_id not in g_infer_context.group_mapping:
                input_len = req.shm_req.input_len
                g_infer_context.group_mapping[group_req_id] = InferReqGroup(
                    group_req_id=group_req_id, share_input_len=input_len, best_of=best_of
                )
            g_infer_context.group_mapping[group_req_id].add_req(r_id)

    def prefill(self, reqs: List[Tuple]):
        req_ids = self._init_reqs(reqs)
        self.build_group(req_ids)
        kwargs, run_req_groups = prepare_prefill_inputs(req_ids, self.is_multimodal)
        logits = self.model.forward(**kwargs)

        next_token_ids, next_token_probs = sample(logits, run_req_groups, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        print(next_token_ids, next_token_logprobs)
        self.post_handel(run_req_groups, next_token_ids, next_token_logprobs)
        return

    def decode(self):
        kwargs, run_req_groups = prepare_decode_inputs(g_infer_context.infer_req_ids)
        logits = self.model.forward(**kwargs)

        next_token_ids, next_token_probs = sample(logits, run_req_groups, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self.post_handel(run_req_groups, next_token_ids, next_token_logprobs)
        return

    def post_handel(self, run_req_groups: List[InferReqGroup], next_token_ids, next_token_logprobs):
        finished_req_ids = []
        start = 0
        for req_group_obj in run_req_groups:
            # prefill and decode is same
            alive_req_id = []
            if req_group_obj.best_of != req_group_obj.refs:
                req_group_obj.update_filter()
            for i in range(req_group_obj.best_of):
                req_obj: InferReq = req_group_obj.get_req(i)
                req_obj.cur_kv_len = req_obj.get_cur_total_len()

                req_obj.set_next_gen_token_id(next_token_ids[start + i], next_token_logprobs[start + i])
                req_obj.cur_output_len += 1

                req_obj.out_token_id_count[next_token_ids[start + i]] += 1
                req_obj.update_finish_status(self.eos_id)
                if req_obj.finish_status.is_finished() or req_obj.shm_req.router_aborted:
                    finished_req_ids.append(req_obj.shm_req.request_id)
                else:
                    alive_req_id.append(req_obj.shm_req.request_id)
                if self.tp_rank < self.dp_size:
                    # shm_cur_kv_len shm_cur_output_len 是 router 调度进程需要读的信息
                    # finish_token_index finish_status candetoken_out_len 是
                    # detokenization 进程需要的信息，注意这些变量的写入顺序避免异步协同问题。
                    req_obj.shm_req.shm_cur_kv_len = req_obj.cur_kv_len
                    req_obj.shm_req.shm_cur_output_len = req_obj.cur_output_len

                    if req_obj.finish_status.is_finished():
                        req_obj.shm_req.finish_token_index = req_obj.get_cur_total_len() - 1
                        req_obj.shm_req.finish_status = req_obj.finish_status

                    req_obj.shm_req.candetoken_out_len = req_obj.cur_output_len

            start += req_group_obj.best_of
            req_group_obj.best_of = len(alive_req_id)
            req_group_obj.req_group = alive_req_id

        g_infer_context.filter(finished_req_ids)
        return
