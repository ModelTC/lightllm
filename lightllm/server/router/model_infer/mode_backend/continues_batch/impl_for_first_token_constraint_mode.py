import os
import shutil
import torch
from .impl import ContinuesBatchBackend
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from .post_process import sample
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FirstTokenConstraintBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        first_allowed_tokens_strs: str = os.environ.get("FIRST_ALLOWED_TOKENS", None)
        logger.info(f"first_allowed_tokens_strs : {first_allowed_tokens_strs}")
        # 使用该模式需要设置FIRST_ALLOWED_TOKENS 环境变量，格式为 "1,2" 或 "1,2,3"  等数字字符串
        assert first_allowed_tokens_strs is not None
        first_allowed_tokens_strs.split(",")
        self.first_allowed_tokens = [int(e.strip()) for e in first_allowed_tokens_strs.split(",") if len(e.strip()) > 0]
        logger.info(f"first_allowed_tokens : {self.first_allowed_tokens}")
        # check token_id < vocab_size
        assert all(e < self.model.vocab_size for e in self.first_allowed_tokens)
        return

    def prefill(self, reqs: List[Tuple]):
        req_ids = self._init_reqs(reqs, init_req_obj=True)
        req_objs = self._trans_req_ids_to_req_objs(req_ids)
        kwargs, run_reqs = prepare_prefill_inputs(req_objs, is_chuncked_mode=False, is_multimodal=self.is_multimodal)
        logits = self.model.forward(**kwargs)
        # first token constraint
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, self.first_allowed_tokens] = False
        logits[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
        )
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )
        assert len(uninit_reqs) == 0
        assert len(prefill_reqs) == 0

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if decode_reqs:
            kwargs, run_reqs = prepare_decode_inputs(decode_reqs)
            logits = self.model.forward(**kwargs)

            self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
