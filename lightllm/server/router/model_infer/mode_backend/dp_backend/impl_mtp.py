import torch
import torch.distributed as dist
import numpy as np
import triton
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq, InferSamplingParams
from lightllm.server.core.objs import FinishStatus
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl_mtp import ContinuesBatchWithMTPBackend

from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
)
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput
from lightllm.common.basemodel.infer_lock import g_infer_state_lock


class DPChunkedPrefillWithMTPBackend(ContinuesBatchWithMTPBackend):
    def __init__(self) -> None:
        super().__init__()
        self.enable_decode_microbatch_overlap = get_env_start_args().enable_decode_microbatch_overlap
        self.enable_prefill_microbatch_overlap = get_env_start_args().enable_prefill_microbatch_overlap
        pass

    def init_model(self, kvargs):
        super().init_model(kvargs)

    def init_custom(self):
        self.reduce_tensor = torch.tensor([0], dtype=torch.int32, device="cuda", requires_grad=False)
        from .pre_process import padded_prepare_prefill_inputs

        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs([], 1, is_multimodal=self.is_multimodal)
        self.model.forward(model_input)
        assert len(run_reqs) == 0 and padded_req_num == 1
        return

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        current_dp_prefill_num = len(prefill_reqs)
        self.reduce_tensor.fill_(current_dp_prefill_num)
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_prefill_num = self.reduce_tensor.item()
        if max_prefill_num != 0:
            if not self.enable_prefill_microbatch_overlap:
                self.normal_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_prefill_reqs(prefill_reqs, max_prefill_num, uninit_reqs, ok_finished_reqs)

        self.reduce_tensor.fill_(len(decode_reqs))
        dist.all_reduce(self.reduce_tensor, op=dist.ReduceOp.MAX, group=None, async_op=False)
        max_decode_num = self.reduce_tensor.item()
        if max_decode_num != 0:
            if not self.enable_decode_microbatch_overlap:
                self.normal_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)
            else:
                self.overlap_decode(decode_reqs, max_decode_num, uninit_reqs, ok_finished_reqs)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def normal_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        from .pre_process import padded_prepare_prefill_inputs

        # main model prefill
        model_input, run_reqs, padded_req_num = padded_prepare_prefill_inputs(
            prefill_reqs, max_prefill_num, is_multimodal=self.is_multimodal
        )
        model_output: ModelOutput = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        next_token_ids_cpu = []
        if len(run_reqs) != 0:
            next_token_ids, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        # spec prefill: MTP
        draft_model_input = model_input
        draft_model_input.hidden_states = model_output.hidden_states

        for draft_model_idx in range(self.spec_step):
            draft_model_input = prepare_mtp_prefill_inputs(
                prefill_reqs,
                model_input,
                next_token_ids_cpu,
                draft_model_idx,
                is_chunked_mode=True,
                padded_req_num=padded_req_num,
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            _, draft_next_token_ids_cpu = self._gen_draft_tokens(draft_model_output)
            model_input.hidden_states = draft_model_output.hidden_states
            self._save_prefill_draft_tokens(draft_next_token_ids_cpu, run_reqs, draft_model_idx)

        if len(run_reqs) != 0:
            self._post_handle(
                run_reqs,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        from .pre_process import padded_prepare_decode_inputs

        model_input, run_reqs, padded_req_num = padded_prepare_decode_inputs(
            decode_reqs, max_decode_num, is_multimodal=self.is_multimodal
        )
        # main model decode
        model_output = self.model.forward(model_input)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        next_token_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
        need_free_mem_indexes = []
        if len(run_reqs) != 0:
            next_token_ids, next_token_probs = sample(model_output.logits[: len(run_reqs)], run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            # verify
            mem_indexes_cpu = model_input.mem_indexes.cpu()
            accepted_reqs, accepted_index, need_free_mem_indexes = self._verify(
                next_token_ids_cpu[: len(run_reqs)], run_reqs, mem_indexes_cpu
            )
            self._post_handle(
                accepted_reqs,
                next_token_ids_cpu[accepted_index],
                next_token_logprobs_cpu[accepted_index],
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

        if padded_req_num != 0:
            next_token_ids = torch.cat(
                [next_token_ids, torch.ones((padded_req_num,), dtype=torch.int64, device="cuda")], dim=0
            )

        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_input.input_ids = next_token_ids
        draft_model_input.hidden_states = model_output.hidden_states
        # process the draft model output
        for draft_model_idx in range(self.spec_step):
            # spec decode: MTP
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            draft_next_token_ids, draft_next_token_ids_cpu = self._gen_draft_tokens(draft_model_output)
            self._save_decode_draft_token_ids(draft_next_token_ids_cpu, run_reqs, draft_model_idx)
            draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.hidden_states = draft_model_output.hidden_states

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        from .pre_process import padded_overlap_prepare_decode_inputs

        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_decode_inputs(decode_reqs, max_decode_num, is_multimodal=self.is_multimodal)
        micro_output, micro_output1 = self.model.microbatch_overlap_decode(micro_input, micro_input1)

        assert micro_output.logits.shape[0] % self.spec_stride == 0
        assert micro_output1.logits.shape[0] % self.spec_stride == 0

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_run_reqs = run_reqs + run_reqs1
        next_token_ids = torch.empty((0,), dtype=torch.int64, device="cuda")
        next_token_ids_cpu = []
        need_free_mem_indexes = []
        if len(all_run_reqs) != 0:
            all_logits = torch.empty(
                (req_num + req_num1, micro_output.logits.shape[1]),
                dtype=micro_output.logits.dtype,
                device=micro_output.logits.device,
            )

            all_logits[0:req_num, :].copy_(micro_output.logits[0:req_num, :], non_blocking=True)
            all_logits[req_num : (req_num + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

            next_token_ids, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()
            micro_mem_indexes_cpu = micro_input.mem_indexes.cpu()
            micro_mem_indexes_cpu1 = micro_input1.mem_indexes.cpu()
            mem_indexes_cpu = torch.cat((micro_mem_indexes_cpu, micro_mem_indexes_cpu1), dim=0)

            # verify
            accepted_reqs, accepted_index, need_free_mem_indexes = self._verify(
                next_token_ids_cpu, all_run_reqs, mem_indexes_cpu
            )

            self._post_handle(
                accepted_reqs,
                next_token_ids_cpu[accepted_index],
                next_token_logprobs_cpu[accepted_index],
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

        # share some inference info with the main model
        draft_micro_input, draft_micro_input1 = micro_input, micro_input1

        draft_micro_input.input_ids = next_token_ids[:req_num]
        draft_micro_input.hidden_states = micro_output.hidden_states
        draft_micro_input1.input_ids = next_token_ids[req_num:]
        draft_micro_input1.hidden_states = micro_output1.hidden_states

        if padded_req_num != 0:
            draft_micro_input.input_ids = torch.cat(
                [draft_micro_input.input_ids, torch.ones((padded_req_num,), dtype=torch.int64, device="cuda")], dim=0
            )
        if padded_req_num1 != 0:
            draft_micro_input1.input_ids = torch.cat(
                [draft_micro_input1.input_ids, torch.ones((padded_req_num1,), dtype=torch.int64, device="cuda")], dim=0
            )

        # process the draft model output
        for draft_model_idx in range(self.spec_step):
            # spec decode: MTP
            draft_micro_output, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_decode(
                draft_micro_input, draft_micro_input1
            )

            draft_next_token_ids, draft_next_token_ids_cpu = self._gen_draft_tokens(draft_micro_output)
            draft_next_token_ids1, draft_next_token_ids_cpu1 = self._gen_draft_tokens(draft_micro_output1)
            self._save_decode_draft_token_ids(draft_next_token_ids_cpu, run_reqs, draft_model_idx)
            self._save_decode_draft_token_ids(draft_next_token_ids_cpu1, run_reqs1, draft_model_idx)
            # prepare inputs for the next draft model
            draft_micro_input.input_ids = draft_next_token_ids
            draft_micro_input.hidden_states = draft_micro_output.hidden_states
            draft_micro_input1.input_ids = draft_next_token_ids1
            draft_micro_input1.hidden_states = draft_micro_output1.hidden_states

        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return

    def overlap_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        from .pre_process import padded_overlap_prepare_prefill_inputs

        (
            micro_input,
            run_reqs,
            padded_req_num,
            micro_input1,
            run_reqs1,
            padded_req_num1,
        ) = padded_overlap_prepare_prefill_inputs(prefill_reqs, max_prefill_num, is_multimodal=self.is_multimodal)

        micro_output, micro_output1 = self.model.microbatch_overlap_prefill(micro_input, micro_input1)

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)

        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_run_reqs = run_reqs + run_reqs1
        next_token_ids_cpu = []
        if len(all_run_reqs) != 0:
            all_logits = torch.empty(
                (len(all_run_reqs), micro_output.logits.shape[1]),
                dtype=micro_output.logits.dtype,
                device=micro_output.logits.device,
            )

            all_logits[0:req_num, :].copy_(micro_output.logits[0:req_num, :], non_blocking=True)
            all_logits[req_num : (req_num + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

            next_token_ids, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

        # spec prefill: MTP
        draft_micro_input, draft_micro_input1 = micro_input, micro_input1
        draft_micro_input.hidden_states = micro_output.hidden_states
        draft_micro_input1.hidden_states = micro_output1.hidden_states

        for draft_model_idx in range(self.spec_step):

            draft_micro_input = prepare_mtp_prefill_inputs(
                run_reqs,
                draft_micro_input,
                next_token_ids_cpu[0:req_num],
                draft_model_idx,
                is_chunked_mode=True,
                padded_req_num=padded_req_num,
            )

            draft_micro_input1 = prepare_mtp_prefill_inputs(
                run_reqs1,
                draft_micro_input1,
                next_token_ids_cpu[req_num:],
                draft_model_idx,
                is_chunked_mode=True,
                padded_req_num=padded_req_num1,
            )

            draft_micro_output, draft_micro_output1 = self.draft_models[draft_model_idx].microbatch_overlap_prefill(
                draft_micro_input, draft_micro_input1
            )
            _, draft_next_token_ids_cpu = self._gen_draft_tokens(draft_micro_output)
            _, draft_next_token_ids_cpu1 = self._gen_draft_tokens(draft_micro_output1)
            self._save_prefill_draft_tokens(draft_next_token_ids_cpu, run_reqs, draft_model_idx)
            self._save_prefill_draft_tokens(draft_next_token_ids_cpu1, run_reqs1, draft_model_idx)

        if len(all_run_reqs) != 0:
            self._post_handle(
                all_run_reqs,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )

        return
