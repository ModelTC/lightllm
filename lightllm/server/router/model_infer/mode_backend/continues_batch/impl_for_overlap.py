import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.log_utils import init_logger
from .overlap_pre_process import overlap_prepare_inputs, overlap_prepare_prefill_inputs, overlap_prepare_decode_inputs
from .post_process import sample

logger = init_logger(__name__)

# 这个模式的gpu使用率很高，但是对首字会有一次decode时间的延迟。
class OverlapBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, finished_reqs, prefill_run_reqs, decode_run_reqs = overlap_prepare_inputs()

        if len(prefill_run_reqs) != 0:
            kwargs = overlap_prepare_prefill_inputs(prefill_run_reqs)
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, prefill_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                prefill_run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
            )

        if len(decode_run_reqs) != 0:
            kwargs = overlap_prepare_decode_inputs(decode_run_reqs)
            logits = self.model.forward(**kwargs)

            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                g_infer_context.filter([req.shm_req.request_id for req in finished_reqs])
                for req in uninit_reqs:
                    req.init_all()
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

            next_token_ids, next_token_probs = sample(logits, decode_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            self._post_handle(
                decode_run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
            )

        else:
            # 利用推理的时间，延迟折叠下一个请求的初始化和退出操作
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                g_infer_context.filter([req.shm_req.request_id for req in finished_reqs])
                for req in uninit_reqs:
                    req.init_all()
            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

        return
