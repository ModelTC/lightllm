import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_chunked_prefill_inputs,
    prepare_draft_main_model_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
import os
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl_mtp import ContinuesBatchWithMTPBackend
import copy
from lightllm.utils.dist_utils import device0_print


logger = init_logger(__name__)


class ChunkedPrefillWithMTPBackend(ContinuesBatchWithMTPBackend):
    def __init__(self) -> None:
        super().__init__()

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if prefill_reqs:
            model_input, run_reqs = prepare_prefill_inputs(
                prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal
            )
            model_output = self.model.forward(model_input)

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            prev_step_has_output = [
                req_obj.get_chuncked_input_token_len() == req_obj.get_cur_total_len() for req_obj in prefill_reqs
            ]
            # spec prefill: MTP
            last_input_ids_cpu = None
            draft_model_input = model_input
            last_hidden_states = model_output.hidden_states
            draft_next_token_ids = next_token_ids
            for draft_model_idx in range(self.spec_step):

                draft_model_input, last_input_ids_cpu, prev_step_has_output = prepare_mtp_chunked_prefill_inputs(
                    prefill_reqs,
                    model_input,
                    last_hidden_states,
                    draft_next_token_ids,
                    draft_model_idx + 1,
                    prev_step_has_output,
                    last_input_ids_cpu,
                )

                draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
                draft_next_token_ids, _ = sample(draft_model_output.logits, run_reqs, self.eos_id)
                draft_next_token_ids = draft_next_token_ids.detach().cpu().numpy()

                last_hidden_states = draft_model_output.hidden_states
                self._save_draft_token_ids(draft_next_token_ids, run_reqs, draft_model_idx)

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
            )

        if decode_reqs:
            model_input, run_reqs, mem_indexes_cpu = prepare_draft_main_model_decode_inputs(
                decode_reqs, self.draft_token_id_map
            )
            model_output = self.model.forward(model_input)
            assert model_output.logits.shape[0] % self.spec_stride == 0

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids_cuda, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids_cuda.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            # verify
            accepted_reqs, accepted_index, need_free_mem_indexes = self.verify(
                next_token_ids, run_reqs, mem_indexes_cpu
            )
            self._post_handle(
                accepted_reqs,
                next_token_ids[accepted_index],
                next_token_logprobs[accepted_index],
                is_chuncked_mode=True,
                do_filter_finished_reqs=False,
            )
            self.main_step += 1

            # share some inference info with the main model
            draft_model_input = model_input
            draft_model_input.input_ids = next_token_ids_cuda
            draft_model_input.hidden_states = model_output.hidden_states
            # process the draft model output
            for draft_model_idx in range(self.spec_step):
                # spec decode: MTP
                draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
                draft_next_token_ids, _ = sample(draft_model_output.logits, run_reqs, self.eos_id)
                # prepare inputs for the next draft model
                draft_model_input.input_ids = draft_next_token_ids
                draft_model_input.hidden_states = draft_model_output.hidden_states
                draft_next_token_ids_numpy = draft_next_token_ids.detach().cpu().numpy()
                self._save_draft_token_ids(draft_next_token_ids_numpy, run_reqs, draft_model_idx)

            if need_free_mem_indexes:
                g_infer_state_lock.acquire()
                g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
                g_infer_state_lock.release()

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
