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
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_chunked_prefill_inputs,
    prepare_draft_main_model_decode_inputs,
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
        model_input, run_reqs = prepare_prefill_inputs([], is_chuncked_mode=True, is_multimodal=self.is_multimodal, pad_for_empty_batch=True)
        self.model.forward(model_input)
        assert len(run_reqs) == 0 and model_input.batch_size == 1
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
        model_input, run_reqs = prepare_prefill_inputs(
            prefill_reqs, is_chuncked_mode=True, is_multimodal=self.is_multimodal, pad_for_empty_batch=True
        )
        model_output: ModelOutput = self.model.forward(model_input)
        
        self._overlap_req_init_and_filter(
            uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
        )
        
        if len(run_reqs) != 0:
            next_token_ids, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            
            prev_step_has_output = [
                req_obj.get_chuncked_input_token_len() == req_obj.get_cur_total_len() for req_obj in prefill_reqs
            ]
        else:
            next_token_ids = np.array([1])
            prev_step_has_output = [True]
            
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
                pad_for_empty_batch=True
            )

            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            last_hidden_states = draft_model_output.hidden_states
            
            if len(run_reqs) != 0:
                draft_next_token_ids, _ = sample(draft_model_output.logits, run_reqs, self.eos_id)
                draft_next_token_ids = draft_next_token_ids.detach().cpu().numpy()
                self._save_draft_token_ids(draft_next_token_ids, run_reqs, draft_model_idx)

        if len(run_reqs) != 0:
            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
            )

    def normal_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        model_input, run_reqs, mem_indexes_cpu = prepare_draft_main_model_decode_inputs(
            decode_reqs, self.draft_token_id_map, pad_for_empty_batch=True
        )
        model_output = self.model.forward(model_input)

        assert len(run_reqs) == 0 or model_output.logits.shape[0] % self.spec_stride == 0

        self._overlap_req_init_and_filter(
            uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
        )

        if len(run_reqs) != 0:
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
        else:
            next_token_ids_cuda = torch.tensor([1], dtype=torch.int64, device='cuda')
            
        self.main_step += 1

        # share some inference info with the main model
        draft_model_input = model_input
        draft_model_input.input_ids = next_token_ids_cuda
        draft_model_input.hidden_states = model_output.hidden_states
        # process the draft model output
        for draft_model_idx in range(self.spec_step):
            # spec decode: MTP
            draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
            if len(run_reqs) != 0:
                draft_next_token_ids, _ = sample(draft_model_output.logits, run_reqs, self.eos_id)
                draft_next_token_ids_numpy = draft_next_token_ids.detach().cpu().numpy()
                self._save_draft_token_ids(draft_next_token_ids_numpy, run_reqs, draft_model_idx)
                draft_model_input.input_ids = draft_next_token_ids
            draft_model_input.hidden_states = draft_model_output.hidden_states

        if len(run_reqs) != 0 and need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()

    def overlap_decode(self, decode_reqs: List[InferReq], max_decode_num: int, uninit_reqs, ok_finished_reqs):
        micro_batch_size = triton.cdiv(max_decode_num, 2)
        micro_batch1_req_num = triton.cdiv(len(decode_reqs), 2)
        micro_input, run_reqs, micro_mem_indexes_cpu = prepare_draft_main_model_decode_inputs(
            decode_reqs[0:micro_batch1_req_num], self.draft_token_id_map, pad_to_tgt_batch_size=micro_batch_size
        )
        micro_input1, run_reqs1, micro_mem_indexes_cpu1 = prepare_draft_main_model_decode_inputs(
            decode_reqs[micro_batch1_req_num:], self.draft_token_id_map, pad_to_tgt_batch_size=micro_batch_size
        )

        micro_output, micro_output1 = self.model.microbatch_overlap_decode(micro_input, micro_input1)
        
        assert micro_output.logits.shape[0] % self.spec_stride == 0
        assert micro_output1.logits.shape[0] % self.spec_stride == 0
        
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        req_num, req_num1 = len(run_reqs), len(run_reqs1)
        all_logits = torch.empty((req_num + req_num1, micro_output.logits.shape[1]), dtype=micro_output.logits.dtype, device=micro_output.logits.device)

        all_logits[0:req_num, :].copy_(micro_output.logits[0:req_num, :], non_blocking=True)
        all_logits[req_num : (req_num + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

        all_run_reqs = run_reqs + run_reqs1
        mem_indexes_cpu = torch.cat((micro_mem_indexes_cpu, micro_mem_indexes_cpu1), dim=0)
        if all_run_reqs:
            next_token_ids_cuda, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids = next_token_ids_cuda.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            
            # verify
            accepted_reqs, accepted_index, need_free_mem_indexes = self.verify(
                next_token_ids, all_run_reqs, mem_indexes_cpu
            )
            
            self._post_handle(
                accepted_reqs, 
                next_token_ids[accepted_index], 
                next_token_logprobs[accepted_index], 
                is_chuncked_mode=True, 
                do_filter_finished_reqs=False
            )
            self.main_step += 1
            
            # share some inference info with the main model
            draft_micro_input, draft_micro_input1 = micro_input, micro_input1
            
            draft_micro_input.input_ids = next_token_ids_cuda[:req_num]
            draft_micro_input.hidden_states = micro_output.hidden_states
            draft_micro_input1.input_ids = next_token_ids_cuda[req_num:]
            draft_micro_input1.hidden_states = micro_output1.hidden_states
            
            all_draft_logits = None
            
            # process the draft model output
            for draft_model_idx in range(self.spec_step):
                # spec decode: MTP
                draft_micro_output, draft_micro_output1 = \
                    self.draft_models[draft_model_idx].microbatch_overlap_decode(draft_micro_input, draft_micro_input1)
                
                if all_draft_logits is None:
                    all_draft_logits = torch.empty((req_num + req_num1, draft_micro_output.logits.shape[1]), 
                        dtype=draft_micro_output.logits.dtype, device=draft_micro_output.logits.device)
                
                all_draft_logits[0:req_num, :].copy_(draft_micro_output.logits[0:req_num, :], non_blocking=True)
                all_draft_logits[req_num : (req_num + req_num1), :].copy_(draft_micro_output1.logits[0:req_num1, :], non_blocking=True)
                
                draft_next_token_ids, _ = sample(all_draft_logits, all_run_reqs, self.eos_id)
                
                # prepare inputs for the next draft model
                draft_micro_input.input_ids = draft_next_token_ids[:req_num]
                draft_micro_input.hidden_states = draft_micro_output.hidden_states
                draft_micro_input1.input_ids = draft_next_token_ids[req_num:]
                draft_micro_input1.hidden_states = draft_micro_output1.hidden_states
                
                draft_next_token_ids_numpy = draft_next_token_ids.detach().cpu().numpy()
                self._save_draft_token_ids(draft_next_token_ids_numpy, all_run_reqs, draft_model_idx)

            if need_free_mem_indexes:
                g_infer_state_lock.acquire()
                g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
                g_infer_state_lock.release()
        return

    def overlap_prefill_reqs(self, prefill_reqs: List[InferReq], max_prefill_num: int, uninit_reqs, ok_finished_reqs):
        micro_batch_size = triton.cdiv(max_prefill_num, 2)
        micro_batch1_req_num = triton.cdiv(len(prefill_reqs), 2)
        
        micro_input0, run_reqs0 = prepare_prefill_inputs(
            prefill_reqs[0:micro_batch1_req_num], is_multimodal=self.is_multimodal, pad_for_empty_batch=True)
        micro_input1, run_reqs1 = prepare_prefill_inputs(
            prefill_reqs[micro_batch1_req_num:], is_multimodal=self.is_multimodal, pad_for_empty_batch=True)
    
        micro_output0, micro_output1 = self.model.microbatch_overlap_prefill(micro_input0, micro_input1)
        
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        
        req_num0, req_num1 = len(run_reqs0), len(run_reqs1)
        all_logits = torch.empty((req_num0 + req_num1, micro_output0.logits.shape[1]), 
                                 dtype=micro_output0.logits.dtype, device=micro_output0.logits.device)

        all_logits[0:req_num0, :].copy_(micro_output0.logits[0:req_num0, :], non_blocking=True)
        all_logits[req_num0 : (req_num0 + req_num1), :].copy_(micro_output1.logits[0:req_num1, :], non_blocking=True)

        all_run_reqs = run_reqs0 + run_reqs1
        if all_run_reqs:
            next_token_ids, next_token_probs = sample(all_logits, all_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            prev_step_has_output0 = [
                req_obj.get_chuncked_input_token_len() == req_obj.get_cur_total_len() for req_obj in all_run_reqs
            ]
            prev_step_has_output1 = [
                req_obj.get_chuncked_input_token_len() == req_obj.get_cur_total_len() for req_obj in all_run_reqs
            ]
            
            # spec prefill: MTP
            last_input_ids_cpu0, last_input_ids_cpu1 = None, None
            draft_micro_input0, draft_micro_input1 = micro_input0, micro_input1
            last_hidden_states0, last_hidden_states1 = micro_output0.hidden_states, micro_output1.hidden_states
            draft_next_token_ids0 = next_token_ids[0:req_num0]
            draft_next_token_ids1 = next_token_ids[req_num0:]
            all_draft_logits = None
            
            for draft_model_idx in range(self.spec_step):
                
                draft_micro_input0, last_input_ids_cpu0, prev_step_has_output0 = prepare_mtp_chunked_prefill_inputs(
                    run_reqs0,
                    draft_micro_input0,
                    last_hidden_states0,
                    draft_next_token_ids0,
                    draft_model_idx + 1,
                    prev_step_has_output0,
                    last_input_ids_cpu0,
                    pad_for_empty_batch=True
                )
                
                draft_micro_input1, last_input_ids_cpu1, prev_step_has_output1 = prepare_mtp_chunked_prefill_inputs(
                    run_reqs1,
                    draft_micro_input1,
                    last_hidden_states1,
                    draft_next_token_ids1,
                    draft_model_idx + 1,
                    prev_step_has_output1,
                    last_input_ids_cpu1,
                    pad_for_empty_batch=True
                )

                draft_micro_output0, draft_micro_output1 = \
                    self.draft_models[draft_model_idx].microbatch_overlap_decode(draft_micro_input0, draft_micro_input1)
                
                if all_draft_logits is None:
                    all_draft_logits = torch.empty((req_num0 + req_num1, draft_micro_output0.logits.shape[1]), 
                        dtype=draft_micro_output0.logits.dtype, device=draft_micro_output0.logits.device)
                
                all_draft_logits[0:req_num0, :].copy_(draft_micro_output0.logits[0:req_num0, :], non_blocking=True)
                all_draft_logits[req_num0 : (req_num0 + req_num1), :].copy_(draft_micro_output1.logits[0:req_num1, :], non_blocking=True)
                
                draft_next_token_ids, _ = sample(all_draft_logits, all_run_reqs, self.eos_id)
                draft_next_token_ids = draft_next_token_ids.detach().cpu().numpy()

                last_hidden_states0 = draft_micro_output0.hidden_states
                last_hidden_states1 = draft_micro_output1.hidden_states
                
                self._save_draft_token_ids(draft_next_token_ids, all_run_reqs, draft_model_idx)

            self._post_handle(
                all_run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=True, do_filter_finished_reqs=False
            )
            
        return
