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
    prepare_mtp_prefill_inputs,
    prepare_draft_main_model_decode_inputs,
    IS_NONE,
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
import os
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.infer_batch import InferReq
import copy
from lightllm.utils.dist_utils import device0_print


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
            }

            mtp_model_cfg, _ = PretrainedConfig.get_config_dict(kvargs["spec_weight_dir"])
            assert mtp_model_cfg["model_type"] == "deepseek_v3"
            assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
            self.draft_models.append(Deepseek3MTPModel(mtp_model_kvargs))

            self.logger.info(f"loaded mtp model class {self.draft_models[i].__class__}")

        max_req_num = kvargs.get("max_req_num", 1000)
        self.draft_token_id_map = torch.full(
            (max_req_num, self.spec_step), fill_value=IS_NONE, dtype=torch.int32, device="cpu"
        )
        self.draft_accepted_len = np.zeros((max_req_num, self.spec_step), dtype=np.int32)
        self.main_step = 0

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
                prefill_reqs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
            )
            model_output = self.model.forward(model_input)

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(model_output.logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            # spec prefill: MTP
            last_input_ids_cpu = None
            draft_model_input = model_input
            last_hidden_states = model_output.hidden_states
            draft_next_token_ids = next_token_ids
            for draft_model_idx in range(self.spec_step):
                draft_model_input, last_input_ids_cpu = prepare_mtp_prefill_inputs(
                    prefill_reqs, model_input, last_hidden_states, draft_next_token_ids, last_input_ids_cpu
                )
                draft_model_output = self.draft_models[draft_model_idx].forward(draft_model_input)
                draft_next_token_ids, _ = sample(draft_model_output.logits, run_reqs, self.eos_id)
                draft_next_token_ids = draft_next_token_ids.detach().cpu().numpy()

                last_hidden_states = draft_model_output.hidden_states
                self._save_draft_token_ids(draft_next_token_ids, run_reqs, draft_model_idx)

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
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
                is_chuncked_mode=False,
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

    def verify(self, next_token_ids, run_reqs, draft_mem_indexes):
        accepted_reqs = []
        accepted_index = []
        need_free_mem_indexes = []
        assert next_token_ids.shape[0] % self.spec_stride == 0

        for i, req in enumerate(run_reqs):
            # main model output
            if i % self.spec_stride == 0:
                accepted_reqs.append(req)
                accepted_index.append(i)
                continue
            draft_model_idx = i % self.spec_stride - 1
            if (
                self.draft_token_id_map[req.req_idx][draft_model_idx] == next_token_ids[i - 1]
                and req.cur_accepted_len == draft_model_idx
            ):
                accepted_reqs.append(req)
                accepted_index.append(i)
                req.cur_accepted_len += 1
            else:
                need_free_mem_indexes.append(draft_mem_indexes[i])
        return accepted_reqs, accepted_index, need_free_mem_indexes

    def _save_draft_token_ids(self, draft_next_token_ids, run_reqs, draft_model_idx):
        batch_size = len(run_reqs) // self.spec_stride
        for i in range(batch_size):
            req = run_reqs[self.spec_stride * i]
            self.draft_token_id_map[req.req_idx][draft_model_idx] = draft_next_token_ids[
                i * self.spec_stride + req.cur_accepted_len
            ]
            #  reset the cur_accepted_len
            if draft_model_idx == self.spec_step - 1:
                if self.is_master_in_dp:
                    req.set_total_accepted_len()
                req.cur_accepted_len = 0
