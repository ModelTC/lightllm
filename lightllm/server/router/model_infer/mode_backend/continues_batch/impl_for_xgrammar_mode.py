import os
import shutil
import torch
from typing import List, Tuple

from .impl import ContinuesBatchBackend
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample

from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        import xgrammar as xgr

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
        self.xgrammar_token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        eos_token_ids = []
        eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.args.eos_id)
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill(self, reqs: List[Tuple]):
        import xgrammar as xgr

        req_ids = self._init_reqs(reqs)
        req_objs = self._trans_req_ids_to_req_objs(req_ids)

        kwargs, run_reqs = prepare_prefill_inputs(req_objs, is_chuncked_mode=False, is_multimodal=self.is_multimodal)

        logics = self.model.forward(**kwargs)

        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_grammar(sample_params.guided_grammar)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            elif sample_params.guided_json is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_json_schema(sample_params.guided_json)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            self._mask_req_out_token(i, run_obj, logics[i])

        # fix the logics with -inf to a large negative value
        logics[logics == float("-inf")] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        self._post_handle(
            run_reqs,
            next_token_ids,
            next_token_logprobs,
            is_chuncked_mode=False,
            do_filter_finished_reqs=False,
            extra_post_req_handle_func=self._update_xgrammer_fsm,
        )

        return

    @calculate_time(show=True, min_cost_ms=200)
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

            all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
            if not all_has_no_constraint:
                for i, run_obj in enumerate(run_reqs):
                    self._mask_req_out_token(i, run_obj, logits[i])

            logits[logits == float("-inf")] = -1000000.0
            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            self._post_handle(
                run_reqs,
                next_token_ids,
                next_token_logprobs,
                is_chuncked_mode=False,
                do_filter_finished_reqs=False,
                extra_post_req_handle_func=self._update_xgrammer_fsm,
            )

        self._overlap_req_init_and_filter(uninit_reqs=[], ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def _update_xgrammer_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        import xgrammar as xgr

        if not hasattr(req_obj.sample_param, "xgrammar_matcher"):
            return

        matcher = req_obj.sampling_param.xgrammar_matcher
        assert matcher.accept_token(next_token_id)
        if matcher.is_terminated():
            req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, logits):
        import xgrammar as xgr

        sample_params = run_obj.sampling_param
        if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
            sample_params.xgrammar_matcher.fill_next_token_bitmask(self.xgrammar_token_bitmask)
            xgr.apply_token_bitmask_inplace(logits, self.xgrammar_token_bitmask.to(logits.device))
        return
