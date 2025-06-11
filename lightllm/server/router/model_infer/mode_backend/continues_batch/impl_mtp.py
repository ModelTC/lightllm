import os
import torch
import numpy as np
from typing import List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.server.router.model_infer.mode_backend.base_backend import ModeBackend
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_prefill_inputs,
    prepare_decode_inputs,
)
from lightllm.server.router.model_infer.mode_backend.mtp_pre_process import (
    prepare_mtp_prefill_inputs,
    IS_NONE,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
from lightllm.server.router.model_infer.infer_batch import InferReq
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelOutput


logger = init_logger(__name__)


class ContinuesBatchWithMTPBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()

    # 支持双模型
    def init_model(self, kvargs):
        super().init_model(kvargs)
        max_total_token_num = self.model.mem_manager.size
        kvargs["max_total_token_num"] = max_total_token_num
        self.spec_step = kvargs.get("spec_step", 1)
        self.spec_stride = self.spec_step + 1
        self.draft_models = []

        os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
        for i in range(self.spec_step):
            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(kvargs["spec_weight_dir"])
            mtp_model_kvargs = {
                "weight_dir": kvargs["spec_weight_dir"],
                "max_total_token_num": kvargs["max_total_token_num"],
                "load_way": kvargs["load_way"],
                "mode": kvargs["mode"],
                "max_req_num": kvargs.get("max_req_num", 1000),
                "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
                "is_token_healing": False,
                "return_all_prompt_logics": False,
                "use_dynamic_prompt_cache": self.use_dynamic_prompt_cache,
                "disable_chunked_prefill": self.disable_chunked_prefill,
                "data_type": kvargs.get("data_type", "float16"),
                "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
                "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
                "disable_cudagraph": kvargs.get("disable_cudagraph", False),
                "mem_fraction": kvargs["mem_fraction"],
                "batch_max_tokens": kvargs.get("batch_max_tokens", None),
                "quant_type": kvargs.get("quant_type", None),
                "quant_cfg": kvargs.get("quant_cfg", None),
                "run_mode": "normal",
                "spec_algo": "MTP_MOUDLE",
                "main_model": self.model,
                "last_mtp_module": i == self.spec_step - 1,
                "mem_layer_start": self.model.config["num_hidden_layers"] + i * mtp_model_cfg["num_hidden_layers"],
            }

            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(kvargs["spec_weight_dir"])
            assert mtp_model_cfg["model_type"] == "deepseek_v3"
            assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
            self.draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))

            self.logger.info(f"loaded mtp model class {self.draft_models[i].__class__}")

    def prefill(self, reqs: List[Tuple]):
        self._init_reqs(reqs, init_req_obj=False)
        return

    def decode(self):
        uninit_reqs, aborted_reqs, ok_finished_reqs, prefill_reqs, decode_reqs = self._get_classed_reqs(
            g_infer_context.infer_req_ids
        )

        if aborted_reqs:
            g_infer_context.filter_reqs(aborted_reqs)

        if prefill_reqs:
            model_input, run_reqs = prepare_prefill_inputs(
                prefill_reqs, is_chuncked_mode=not self.disable_chunked_prefill, is_multimodal=self.is_multimodal
            )
            model_output = self.model.forward(model_input)

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
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
                    is_chunked_mode=not self.disable_chunked_prefill,
                )
                draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
                _, draft_next_token_ids_cpu = self._gen_draft_tokens(draft_model_output)
                model_input.hidden_states = draft_model_output.hidden_states
                self._save_prefill_draft_tokens(draft_next_token_ids_cpu, run_reqs, draft_model_idx)

            self._post_handle(
                run_reqs,
                next_token_ids_cpu,
                next_token_logprobs_cpu,
                is_chuncked_mode=not self.disable_chunked_prefill,
                do_filter_finished_reqs=False,
            )

        if decode_reqs:
            model_input, run_reqs = prepare_decode_inputs(decode_reqs)
            model_output = self.model.forward(model_input)
            assert model_output.logits.shape[0] % self.spec_stride == 0

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
            next_token_ids_cpu = next_token_ids.detach().cpu().numpy()
            next_token_logprobs_cpu = torch.log(next_token_probs).detach().cpu().numpy()

            # verify
            mem_indexes_cpu = model_input.mem_indexes.cpu()
            accepted_reqs, accepted_index, need_free_mem_indexes = self._verify(
                next_token_ids_cpu, run_reqs, mem_indexes_cpu
            )
            self._post_handle(
                accepted_reqs,
                next_token_ids_cpu[accepted_index],
                next_token_logprobs_cpu[accepted_index],
                is_chuncked_mode=not self.disable_chunked_prefill,
                do_filter_finished_reqs=False,
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
                # prepare inputs for the next draft model
                draft_model_input.input_ids = draft_next_token_ids
                draft_model_input.hidden_states = draft_model_output.hidden_states
                self._save_decode_draft_token_ids(draft_next_token_ids_cpu, run_reqs, draft_model_idx)

            if need_free_mem_indexes:
                g_infer_state_lock.acquire()
                g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
                g_infer_state_lock.release()

        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return

    def _gen_draft_tokens(self, model_output: ModelOutput):
        logits = model_output.logits
        probs = torch.softmax(logits, dim=-1)
        draft_next_token_ids = torch.argmax(probs, dim=-1)
        return draft_next_token_ids, draft_next_token_ids.detach().cpu().numpy()

    def _verify(self, next_token_ids: torch.Tensor, run_reqs: List[InferReq], draft_mem_indexes: torch.Tensor):
        accepted_reqs = []
        accepted_index = []
        need_free_mem_indexes = []
        assert next_token_ids.shape[0] % self.spec_stride == 0
        batch_size = next_token_ids.shape[0] // self.spec_stride
        for b in range(batch_size):
            req = run_reqs[b % self.spec_stride]
            req_start_idx = b * self.spec_stride
            req_end_idx = (b + 1) * self.spec_stride
            # step_idx==0 means the output of the main model
            for step_idx in range(self.spec_stride):
                if step_idx == 0 or req.mtp_gen_token_ids[step_idx - 1] == next_token_ids[req_start_idx + step_idx - 1]:
                    accepted_reqs.append(req)
                    accepted_index.append(req_start_idx + step_idx)
                    req.mtp_cur_accepted_len += 1 if step_idx != 0 else 0
                else:
                    need_free_mem_indexes.extend(draft_mem_indexes[req_start_idx + step_idx : req_end_idx])
                    break
            #  reset the mtp status
            req.mtp_gen_token_ids = []
        return accepted_reqs, accepted_index, need_free_mem_indexes

    def _save_prefill_draft_tokens(
        self, draft_next_token_ids: torch.Tensor, run_reqs: List[InferReq], draft_model_idx: int
    ):
        batch_size = len(run_reqs)
        for i in range(batch_size):
            req = run_reqs[i]
            # if the request has unfinished chunked tokens, skip it.
            if req.get_chuncked_input_token_len() < req.get_cur_total_len():
                continue
            req.mtp_gen_token_ids.append(draft_next_token_ids[i])

    def _save_decode_draft_token_ids(
        self, draft_next_token_ids: torch.Tensor, run_reqs: List[InferReq], draft_model_idx: int
    ):
        batch_size = len(run_reqs) // self.spec_stride
        for i in range(batch_size):
            req = run_reqs[self.spec_stride * i]
            # append the draft token
            req.mtp_gen_token_ids.append(draft_next_token_ids[i * self.spec_stride + req.mtp_cur_accepted_len])
            #  reset the mtp status
            if draft_model_idx == self.spec_step - 1:
                if self.is_master_in_dp:
                    req.set_total_accepted_len()
                req.mtp_cur_accepted_len = 0
