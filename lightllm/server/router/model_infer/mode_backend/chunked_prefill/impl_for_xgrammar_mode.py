import torch
from typing import List, Optional, Tuple, Union, Dict
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
)


from .impl import ChunkedPrefillBackend
from lightllm.server.router.model_infer.mode_backend.pre import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.chunked_prefill.triton_ops.bit_mask_ops import (
    apply_token_bitmask_inplace_triton,
)
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.core.objs import FinishStatus
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

MAX_ROLLBACK_TOKENS = 200


class XGrammarGrammar:
    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]] = None,
        key_string: Optional[str] = None,
    ) -> None:
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.finished = False
        self.accepted_tokens = []
        self.key_string = key_string

    def accept_token(self, token: int):
        if not self.is_terminated():
            accepted = self.matcher.accept_token(token)
            if not accepted:
                # log for debugging
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Key string: {self.key_string}"
                )
            else:
                self.accepted_tokens.append(token)
            return accepted

    def rollback(self, k: int):
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def is_terminated(self):
        return self.matcher.is_terminated()

    def allocate_vocab_mask(self, vocab_size: int, batch_size: int, device) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        if logits.device.type == "cuda":
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
        elif logits.device.type == "cpu" and self.apply_vocab_mask_cpu:
            self.apply_vocab_mask_cpu(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    def copy(self):
        matcher = GrammarMatcher(
            self.ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(self, old_output_ids: List[int], new_output_ids: List[int], next_state: int):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=})"


class XgrammarBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.cache: Dict[Tuple[str, str], XGrammarGrammar] = {}

    def init_custom(self):
        import xgrammar as xgr

        self.tokenizer = get_tokenizer(
            self.args.model_dir, self.args.tokenizer_mode, trust_remote_code=self.args.trust_remote_code
        )

        self.tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
        self.xgrammar_compiler = xgr.GrammarCompiler(self.tokenizer_info, max_threads=8)
        self.vocab_mask = None
        self.vocab_size = self.tokenizer_info.vocab_size

        eos_token_ids = []
        eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.args.eos_id)
        return

    @calculate_time(show=False, min_cost_ms=300)
    def decode(self):

        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        # 先 decode
        if decode_reqs:
            model_input, run_reqs = prepare_decode_inputs(decode_reqs)
            model_output = self.model.forward(model_input)
            logits = model_output.logits
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            self._init_req_xgrammer_matcher_infos(run_reqs=run_reqs)
            all_has_no_constraint = all([not e.sampling_param.has_constraint_setting() for e in run_reqs])
            first_grammar = None
            if not all_has_no_constraint:
                for i, run_obj in enumerate(run_reqs):
                    if (
                        run_obj.sampling_param.guided_grammar is not None
                        or run_obj.sampling_param.guided_json is not None
                    ):
                        if first_grammar is None:
                            first_grammar = run_obj.sampling_param.guided_grammar or run_obj.sampling_param.guided_json
                    self._mask_req_out_token(i, run_obj, logits[i])
            if first_grammar is not None:
                self.vocab_mask = XGrammarGrammar.move_vocab_mask(self.vocab_mask, logits.device)
                apply_token_bitmask_inplace_triton(logits, self.vocab_mask)
                logits[logits == float("-inf")] = -1000000.0
            # mask out the padding token logits
            logits[:, self.tokenizer_info.vocab_size :] = -1000000.0

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
            del model_output
            del logits

        # 再 prefill
        if len(decode_reqs) == 0 or (self.forward_step % self.max_wait_step == 0) or (self.need_prefill_count > 0):
            if prefill_reqs:
                self.need_prefill_count -= 1
                model_input, run_reqs = prepare_prefill_inputs(
                    prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
                )
                model_output = self.model.forward(model_input)
                logits = model_output.logits
                self._overlap_req_init_and_filter(
                    uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
                )

                first_grammar = None
                self._init_req_xgrammer_matcher_infos(run_reqs=run_reqs)
                for i, run_obj in enumerate(run_reqs):
                    if (
                        run_obj.sampling_param.guided_grammar is not None
                        or run_obj.sampling_param.guided_json is not None
                    ):
                        if first_grammar is None:
                            first_grammar = run_obj.sampling_param.guided_grammar or run_obj.sampling_param.guided_json
                    self._mask_req_out_token(i, run_obj, logits[i])

                if first_grammar is not None:
                    self.vocab_mask = XGrammarGrammar.move_vocab_mask(self.vocab_mask, logits.device)
                    # fix the logics with -inf to a large negative value
                    apply_token_bitmask_inplace_triton(logits, self.vocab_mask)
                    logits[logits == float("-inf")] = -1000000.0
                # mask out the padding token logits
                logits[:, self.tokenizer_info.vocab_size :] = -1000000.0
                next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
                next_token_ids = next_token_ids.detach().cpu().numpy()
                next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
                self._post_handle(
                    run_reqs,
                    next_token_ids,
                    next_token_logprobs,
                    is_chuncked_mode=True,
                    do_filter_finished_reqs=False,
                    extra_post_req_handle_func=self._update_xgrammer_fsm,
                )
                del model_output
                del logits

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        self.forward_step += 1
        return

    def _update_xgrammer_fsm(self, req_obj: InferReq, next_token_id, next_token_logprob):
        import xgrammar as xgr

        if not hasattr(req_obj.sampling_param, "xgrammar_matcher"):
            return

        matcher = req_obj.sampling_param.xgrammar_matcher
        assert matcher.accept_token(next_token_id)
        if matcher.is_terminated():
            req_obj.finish_status.set_status(FinishStatus.FINISHED_STOP)
        return

    def _mask_req_out_token(self, i, run_obj: InferReq, logits):
        import xgrammar as xgr

        if run_obj.get_chuncked_input_token_len() == run_obj.get_cur_total_len():
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None or sample_params.guided_json is not None:
                sample_params.xgrammar_matcher.fill_vocab_mask(self.vocab_mask, i)
        return

    def _init_req_xgrammer_matcher_infos(self, run_reqs: List[InferReq]):
        import xgrammar as xgr

        self.vocab_mask = xgr.allocate_token_bitmask(
            batch_size=len(run_reqs),
            vocab_size=self.tokenizer_info.vocab_size,
        )
        for i, run_obj in enumerate(run_reqs):
            run_obj: InferReq = run_obj
            sample_params = run_obj.sampling_param
            if sample_params.guided_grammar is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    key = ("ebnf", sample_params.guided_grammar)
                    value = self.cache.get(key)
                    if value:
                        sample_params.xgrammar_matcher = value.copy()
                    else:
                        ctx = self.xgrammar_compiler.compile_grammar(sample_params.guided_grammar)
                        matcher = GrammarMatcher(
                            ctx,
                            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
                        )
                        self.cache[key] = XGrammarGrammar(matcher, self.vocab_size, ctx)
                        sample_params.xgrammar_matcher = self.cache[key].copy()
            elif sample_params.guided_json is not None:
                if not hasattr(sample_params, "xgrammar_matcher"):
                    key = ("json", sample_params.guided_grammar)
                    value = self.cache.get(key)
                    if value:
                        sample_params.xgrammar_matcher = value.copy()
                    else:
                        ctx = self.xgrammar_compiler.compile_json_schema(sample_params.guided_json)
                        matcher = GrammarMatcher(
                            ctx,
                            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
                        )
                        self.cache[key] = XGrammarGrammar(matcher, self.vocab_size, ctx)
                        sample_params.xgrammar_matcher = self.cache[key].copy()
        return
