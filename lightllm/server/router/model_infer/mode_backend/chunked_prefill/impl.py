import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.utils.envs_utils import get_env_start_args
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from ..continues_batch.post_process import sample


logger = init_logger(__name__)


class ChunkedPrefillBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.forward_step = 0
        args = get_env_start_args()
        self.max_wait_step = args.router_max_wait_tokens

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs)
        self.forward_step = 0  # prefill first
        return

    def decode(self):
        kwargs, run_reqs = prepare_decode_inputs(g_infer_context.infer_req_ids)
        self.forward_batch(kwargs, run_reqs)
        if len(run_reqs) == 0 or self.forward_step % self.max_wait_step == 0:
            # run prefill
            kwargs, run_reqs = prepare_prefill_inputs(g_infer_context.infer_req_ids)
            self.forward_batch(kwargs, run_reqs)
        self.forward_step += 1
        return

    def forward_batch(self, kwargs, run_reqs):
        if len(run_reqs) == 0:
            return
        logits = self.model.forward(**kwargs)
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        self._post_handle(
            run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=True
        )
        return
