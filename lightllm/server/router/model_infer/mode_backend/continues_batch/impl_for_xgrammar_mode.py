import json
import os
import shutil
import torch
import xgrammar as xgr

from .impl import ContinuesBatchBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.io_struct import FinishStatus
from lightllm.server.router.model_infer.infer_batch import InferBatch, InferReq, InferSamplingParams
from .pre_process import prepare_prefill_inputs, prepare_decode_inputs
from .post_process import sample
from lightllm.server.tokenizer import get_tokenizer
from typing import List
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class XgrammarBackend(ContinuesBatchBackend):
    def __init__(self) -> None:
        super().__init__()

    def init_custom(self):
        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=8)
        self.xgrammar_token_bitmask = xgr.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        eos_token_ids = []
        eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.args.eos_id)
        # self.tokenizer.eos_token_ids = eos_token_ids
        # logger.info(f"eos_ids {self.tokenizer.eos_token_ids}")
        return

    @calculate_time(show=False, min_cost_ms=300)
    def prefill_batch(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_prefill_inputs(batch, self.radix_cache, self.model.mem_manager)
        run_reqs: List[InferReq] = run_reqs

        logics = self.model.forward(**kwargs)

        mask = torch.ones_like(logics, dtype=torch.bool)
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_grammar(sample_params.guided_grammar)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            elif sample_params.guided_json is not None:
                xgrammar_compiled_grammar = self.xgrammar_compiler.compile_json_schema(sample_params.guided_json)
                sample_params.xgrammar_matcher = xgr.GrammarMatcher(xgrammar_compiled_grammar)
            self._mask_req_out_token(i, run_obj, mask, logics[i])

        # fix the logics with -inf to a large negative value
        logics[logics == float("-inf")] = -1000000.0
        logics[mask] = -1000000.0

        next_token_ids, next_token_probs = sample(logics, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    @calculate_time(show=True, min_cost_ms=200)
    def decode_batch(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs = prepare_decode_inputs(batch, self.radix_cache)
        run_reqs: List[InferReq] = run_reqs

        logits = self.model.forward(**kwargs)

        all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
        if not all_has_no_constraint:
            mask = torch.ones_like(logits, dtype=torch.bool)
            for i, run_obj in enumerate(run_reqs):
                self._mask_req_out_token(i, run_obj, mask, logits[i])
            logits[mask] = -1000000.0

        logits[logits == float("-inf")] = -1000000.0
        next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

        for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
            req_obj: InferReq = req_obj
            req_obj.cur_kv_len = len(req_obj.input_token_ids)
            req_obj.input_token_ids.append(next_token_id)
            req_obj.out_token_id_count[next_token_id] += 1
            req_obj.update_finish_status(self.eos_id)

            self._handle_req_ans(req_obj, next_token_id, next_token_logprob, output_dict)

        self.cache[batch.batch_id] = batch
        return output_dict

    def _handle_req_ans(self, req_obj: InferReq, next_token_id, next_token_logprob, output_dict):
        next_token_id = int(next_token_id)
        if req_obj.sampling_param.guided_grammar is not None or req_obj.sampling_param.guided_json is not None:
            sample_params = req_obj.sampling_param
            if sample_params.xgrammar_matcher.is_terminated():
                req_obj.finish_status = FinishStatus.FINISHED_STOP
            else:
                assert sample_params.xgrammar_matcher.accept_token(next_token_id)

        metadata = {
            "id": next_token_id,
            "logprob": float(next_token_logprob),
        }
        output_dict[req_obj.r_id] = (
            req_obj.req_status,
            req_obj.cur_kv_len,
            req_obj.get_output_len(),
            [(next_token_id, metadata)],
            req_obj.finish_status.value,
            None,
        )
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, mask, logits):
        sample_params = run_obj.sampling_param
        if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
            sample_params.xgrammar_matcher.fill_next_token_bitmask(self.xgrammar_token_bitmask)
            xgr.apply_token_bitmask_inplace(logits, self.xgrammar_token_bitmask.to(logits.device))
            mask[i, :] = False
        elif sample_params.allowed_token_ids is not None:
            mask[i, sample_params.allowed_token_ids] = False
        else:
            mask[i, :] = False
        return
