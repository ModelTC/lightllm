import torch
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
    prepare_mtp_main_model_decode_inputs,
    IS_NONE
)
from lightllm.server.router.model_infer.mode_backend.generic_post_process import sample
import os
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
import numpy as np
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.server.router.model_infer.infer_batch import InferReq
import copy
from lightllm.utils.custom_kernel_utis import custom_cat


logger = init_logger(__name__)

# TODO: optim
def update_draft_token_mem_indexes(draft_token_memindex_map, run_reqs, mem_indexes):
    for i, req in enumerate(run_reqs):
        draft_token_memindex_map[req.idx] = mem_indexes[i]

class ContinuesBatchWithMTPBackend(ModeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.accepted_cnt = 0
        self.all_cnt = 0
        
    # 支持双模型
    def init_model(self, kvargs): 
        mem_fraction = kvargs.get("mem_fraction", 0.9)
        kvargs["mem_fraction"] = mem_fraction * 0.9 # TODO
        super().init_model(kvargs)
        
        os.environ["DISABLE_CHECK_MAX_LEN_INFER"] = "1"
        mtp_model_kvargs = {
            "weight_dir": kvargs["spec_weight_dir"],
            "max_total_token_num": kvargs["max_total_token_num"],
            "load_way": kvargs["load_way"],
            "mode": kvargs["mode"],
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "is_token_healing": False,
            "return_all_prompt_logics": False,
            "use_dynamic_prompt_cache": False,
            "disable_chunked_prefill": True, # TODO
            "data_type": kvargs.get("data_type", "float16"),
            "graph_max_batch_size": kvargs.get("graph_max_batch_size", 16),
            "graph_max_len_in_batch": kvargs.get("graph_max_len_in_batch", 8196),
            "disable_cudagraph": kvargs.get("disable_cudagraph", False),
            "mem_fraction": mem_fraction,
            "batch_max_tokens": kvargs.get("batch_max_tokens", None),
            "quant_type": kvargs.get("quant_type", None),
            "quant_cfg": kvargs.get("quant_cfg", None),
            "run_mode": "normal",
            "spec_algo": "MTP_MOUDLE",
        }

        mtp_model_cfg, _ = PretrainedConfig.get_config_dict(kvargs["spec_weight_dir"])
        assert mtp_model_cfg["model_type"] == "deepseek_v3"
        assert mtp_model_cfg["architectures"][0] == "DeepseekV3ForCausalLMNextN"
        self.mtp_model = Deepseek3MTPModel(mtp_model_kvargs)
        
        # shared weight
        self.mtp_model.pre_post_weight.wte_weight_ = self.model.pre_post_weight.wte_weight_
        self.mtp_model.pre_post_weight.lm_head_weight_ = self.model.pre_post_weight.lm_head_weight_
        self.mtp_model.pre_post_weight.final_norm_weight_ = self.model.pre_post_weight.final_norm_weight_

        self.logger.info(f"loaded mtp model class {self.mtp_model.__class__}")
        
        max_req_num = kvargs.get("max_req_num", 1000)
        self.draft_token_id_map = torch.full((max_req_num,), fill_value=IS_NONE, dtype=torch.int32, device="cuda") 
        self.main_draft_token_memindex_map = torch.full((max_req_num,), fill_value=IS_NONE, dtype=torch.int32, device="cuda")
        self.mtp_draft_token_memindex_map = torch.full((max_req_num,), fill_value=IS_NONE, dtype=torch.int32, device="cuda")
        
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
            kwargs, run_reqs = prepare_prefill_inputs(
                prefill_reqs, is_chuncked_mode=False, is_multimodal=self.is_multimodal
            )
            logits = self.model.forward(**kwargs)

            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            next_token_ids, next_token_probs = sample(logits, run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            # spec decode: MTP
            self.mtp_model.spec_info = self.model.spec_info
            mtp_kwargs = prepare_mtp_prefill_inputs(prefill_reqs, next_token_ids, self.mtp_model.mem_manager)
            draft_logits = self.mtp_model.forward(**mtp_kwargs)
            draft_next_token_ids, _ = sample(draft_logits, run_reqs, self.eos_id)
            draft_next_token_ids = draft_next_token_ids.detach().cpu().numpy()
            self._save_draft_token_ids(draft_next_token_ids, run_reqs)

            self._post_handle(
                run_reqs, next_token_ids, next_token_logprobs, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

        if decode_reqs:
            kwargs, run_reqs = prepare_mtp_main_model_decode_inputs(decode_reqs, self.draft_token_id_map)
            update_draft_token_mem_indexes(self.main_draft_token_memindex_map, run_reqs, kwargs["mem_indexes"][1::2])
            logits = self.model.forward(**kwargs)
            assert logits.shape[0] % 2 == 0
            
            self._overlap_req_init_and_filter(
                uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True
            )

            twist_run_reqs  = [item for item in run_reqs for _ in range(2)]
            next_token_ids, next_token_probs = sample(logits, twist_run_reqs, self.eos_id)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            next_token_ids0 = next_token_ids[::2]
            next_token_logprobs0 = next_token_logprobs[::2]
            self._post_handle(
                run_reqs, next_token_ids0, next_token_logprobs0, is_chuncked_mode=False, do_filter_finished_reqs=False
            )
            
            next_token_ids1 = next_token_ids[1::2]
            next_token_logprobs1 = next_token_logprobs[1::2]
            
            accepted_reqs =  self.verify(next_token_ids1, run_reqs)
            self._post_handle(
                accepted_reqs, next_token_ids1, next_token_logprobs1, is_chuncked_mode=False, do_filter_finished_reqs=False
            )

            # spec decode: MTP
            self.mtp_model.spec_info = self.model.spec_info
            mtp_kwargs = copy.deepcopy(kwargs)
            mtp_kwargs["input_ids"] = torch.tensor(next_token_ids, dtype=torch.int64, device="cuda")
            mtp_mem_indexes = self.mtp_model.mem_manager.alloc(next_token_ids.shape[0]).cuda()
            kwargs['mem_indexes'] = mtp_mem_indexes
            update_draft_token_mem_indexes(self.mtp_draft_token_memindex_map, run_reqs, mtp_mem_indexes[1::2])
            
            draft_logits = self.mtp_model.forward(**mtp_kwargs)
            draft_next_token_ids, _ = sample(draft_logits, twist_run_reqs, self.eos_id)

            accepted_req_idxs = [req.req_idx for req in accepted_reqs]
            self._save_draft_token_ids(draft_next_token_ids, run_reqs, accepted_req_idxs)

            
        self._overlap_req_init_and_filter(uninit_reqs=uninit_reqs, ok_finished_reqs=ok_finished_reqs, clear_list=True)
        return
    
    def verify(self, next_token_ids1, run_reqs):
        accepted_reqs = []
        need_free_mem_indexes = []
        for i, req in enumerate(run_reqs):
            if self.main_draft_token_memindex_map[req.req_idx] == next_token_ids1[i]:
                self.accepted_map[req.req_idx] = True
                accepted_reqs.append(req)
                self.main_draft_token_memindex_map[req.req_idx] = IS_NONE
            else:
                self.accepted_map[req.req_idx] = False
                need_free_mem_indexes.append(self.main_draft_token_memindex_map[req.req_idx])
        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            g_infer_context.req_manager.mem_manager.free(need_free_mem_indexes)
            g_infer_state_lock.release()
        return accepted_reqs
    
    def _save_draft_token_ids(self, draft_next_token_ids, run_reqs, accepted_reqs=None):
        assert accepted_reqs is None or draft_next_token_ids.shape[0] == 2 * len(run_reqs)
        need_free_mem_indexes = []
        for i, req in enumerate(run_reqs):
            if accepted_reqs is None:
                self.mtp_draft_token_memindex_map[req.req_idx] = draft_next_token_ids[i]  
            else:
                if req.req_idx in accepted_reqs:
                    self.draft_token_map[req.req_idx] = draft_next_token_ids[2*i+1]
                    self.mtp_draft_token_memindex_map[req.req_idx] = IS_NONE     
                else:
                    self.draft_token_map[req.req_idx] = draft_next_token_ids[2*i]
                    need_free_mem_indexes.append(self.mtp_draft_token_memindex_map[req.req_idx])
        if need_free_mem_indexes:
            g_infer_state_lock.acquire()
            self.mtp_model.req_manager.free([], need_free_mem_indexes)
            g_infer_state_lock.release()
        return
    
    def _overlap_req_init_and_filter(
        self, uninit_reqs: List[InferReq], ok_finished_reqs: List[InferReq], clear_list=False
    ):
        if uninit_reqs or ok_finished_reqs:
            with torch.cuda.stream(g_infer_context.get_overlap_stream()):
                if ok_finished_reqs:
                    g_infer_state_lock.acquire()
                    self._free_mtp_model_memindex(ok_finished_reqs)
                    g_infer_context.filter_reqs(ok_finished_reqs)
                    g_infer_state_lock.release()

                if uninit_reqs:
                    g_infer_state_lock.acquire()
                    self._post_init_reqs(uninit_reqs)
                    g_infer_state_lock.release()

            torch.cuda.current_stream().wait_stream(g_infer_context.get_overlap_stream())

            if clear_list:
                uninit_reqs.clear()
                ok_finished_reqs.clear()
        return
    
    def _free_mtp_model_memindex(self, ok_finished_reqs):
        mtp_free_mem_indexes = []
        for req in ok_finished_reqs:
            mtp_free_mem_indexes.append(
                self.mtp_model.req_manager.req_to_token_indexs[req.req_idx][req.shm_req.input_len : req.cur_kv_len]
            )
        free_memindexes = custom_cat(mtp_free_mem_indexes)
        g_infer_state_lock.acquire()
        g_infer_context.req_manager.mem_manager.free(free_memindexes)
        g_infer_state_lock.release()