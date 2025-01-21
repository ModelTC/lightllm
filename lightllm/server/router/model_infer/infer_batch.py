import os
import copy
import time
import torch
import torch.distributed as dist
import numpy as np
import collections

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from lightllm.common.req_manager import ReqManager
from lightllm.utils.infer_utils import mark_start, mark_end
from lightllm.server.core.objs import Req, SamplingParams, FinishStatus, ShmReqManager
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.utils.log_utils import init_logger
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.common.basemodel.infer_lock import g_infer_state_lock

logger = init_logger(__name__)


@dataclass
class InferenceContext:
    req_manager: ReqManager = None  # gpu 请求管理
    radix_cache: RadixCache = None
    shm_req_manager: ShmReqManager = None  # 共享内存请求对象管理
    requests_mapping: Dict[int, "InferReq"] = None
    group_mapping = None  # 只有进行多输出模式下才有真的使用
    infer_req_ids = None
    vocab_size = None

    def register(
        self, req_manager: ReqManager, radix_cache: RadixCache, shm_req_manager: ShmReqManager, vocab_size: int
    ):
        self.req_manager = req_manager
        self.radix_cache = radix_cache
        self.shm_req_manager = shm_req_manager

        self.requests_mapping = {}
        self.group_mapping = {}
        self.infer_req_ids = []

        self.vocab_size = vocab_size
        return

    def add_reqs(self, requests: List[Tuple[int, int, Any, int]], init_req_obj=True):
        request_ids = []
        for r in requests:

            r_id, r_index, multimodal_params, _ = r
            if r_id not in self.requests_mapping.keys():
                shm_req = self.shm_req_manager.get_req_obj_by_index(r_index)
                r_obj = InferReq(
                    shm_req=shm_req,
                    req_idx=self.req_manager.alloc(),
                    sampling_param=InferSamplingParams(shm_req, self.vocab_size),
                    multimodal_params=multimodal_params,
                )
                self.requests_mapping[r_id] = r_obj
            else:
                r_obj: InferReq = self.requests_mapping[r_id]

            request_ids.append(r_id)

            if init_req_obj:
                r_obj.init_all()

        self.infer_req_ids.extend(request_ids)

        return

    def free_a_req_mem(self, free_token_index: List, req: "InferReq", is_group_finished: bool):
        if self.radix_cache is None:
            free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len])
        else:
            input_token_ids = req.get_input_token_ids()
            key = torch.tensor(input_token_ids[0 : req.cur_kv_len], dtype=torch.int64, device="cpu")
            value = self.req_manager.req_to_token_indexs[req.req_idx][: req.cur_kv_len].detach().cpu()
            if is_group_finished:
                prefix_len = self.radix_cache.insert(key, value)
                old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
                free_token_index.append(self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len:prefix_len])
                if req.shared_kv_node is not None:
                    assert req.shared_kv_node.node_prefix_total_len <= prefix_len
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None
            else:
                old_prefix_len = 0 if req.shared_kv_node is None else req.shared_kv_node.node_prefix_total_len
                free_token_index.append(
                    self.req_manager.req_to_token_indexs[req.req_idx][old_prefix_len : req.cur_kv_len]
                )
                if req.shared_kv_node is not None:
                    self.radix_cache.dec_node_ref_counter(req.shared_kv_node)
                    req.shared_kv_node = None

    @torch.no_grad()
    def filter(self, finished_request_ids: List[int]):
        if len(finished_request_ids) == 0:
            return

        free_req_index = []
        free_token_index = []
        for request_id in finished_request_ids:
            req: InferReq = self.requests_mapping.pop(request_id)
            group_req_id = convert_sub_id_to_group_id(req.shm_req.request_id)
            if group_req_id in self.group_mapping:
                is_group_finished = self.group_mapping[group_req_id].decrease_refs(req.shm_req.request_id)
                if is_group_finished:
                    del self.group_mapping[group_req_id]
                self.free_a_req_mem(free_token_index, req, is_group_finished)
            else:
                self.free_a_req_mem(free_token_index, req, True)
            free_req_index.append(req.req_idx)
            # logger.info(f"infer release req id {req.shm_req.request_id}")
            req.shm_req.shm_infer_released = True
            self.shm_req_manager.put_back_req_obj(req.shm_req)

        free_token_index = torch.cat(free_token_index, dim=-1)
        self.req_manager.free(free_req_index, free_token_index)

        finished_req_ids_set = set(finished_request_ids)
        self.infer_req_ids = [_id for _id in self.infer_req_ids if _id not in finished_req_ids_set]

        if self.radix_cache is not None and len(self.infer_req_ids) == 0:
            logger.debug(
                f"free a batch state:\n"
                f"radix refed token num {self.radix_cache.get_refed_tokens_num()}\n"
                f"radix hold token num {self.radix_cache.get_tree_total_tokens_num()}\n"
                f"mem manager can alloc token num {self.req_manager.mem_manager.can_use_mem_size}\n"
                f"mem manager total size {self.req_manager.mem_manager.size}"
            )

        return

    @torch.no_grad()
    def pause_reqs(self, pause_req_ids: List[int]):
        free_token_index = []
        for request_id in pause_req_ids:
            req: InferReq = self.requests_mapping[request_id]
            self.infer_req_ids.remove(request_id)

            # 不支持多输出的情况的暂停
            self.free_a_req_mem(free_token_index, req, is_group_finished=True)
            req.cur_kv_len = 0
            req.shm_req.shm_cur_kv_len = req.cur_kv_len

        if len(free_token_index) != 0:
            free_token_index = torch.cat(free_token_index, dim=-1)
            self.req_manager.free_token(free_token_index)

        return self


g_infer_context = InferenceContext()


class InferSamplingParams:
    def __init__(
        self,
        shm_req: Req,
        vocab_size: int,
    ) -> None:
        self.shm_param = shm_req.sample_params
        if self.shm_param.top_k == -1:
            self.shm_param.top_k = vocab_size

        # output constraint states
        self.regular_constraint = self.shm_param.regular_constraint.to_str()
        if len(self.regular_constraint) == 0:
            self.regular_constraint = None

        self.regex_guide = None
        self.fsm_current_state: int = 0
        self.allowed_token_ids = self.shm_param.allowed_token_ids.to_list()
        if len(self.allowed_token_ids) == 0:
            self.allowed_token_ids = None

        # p d mode use params
        if self.shm_param.move_kv_to_decode_node.exists:
            self.move_kv_to_decode_node = self.shm_param.move_kv_to_decode_node.to_dict()
        else:
            self.move_kv_to_decode_node = None

        # this check is not very good to placed here. to do...
        if self.allowed_token_ids is not None:
            if not all(e < vocab_size for e in self.allowed_token_ids):
                logger.error("allowed_token_ids contain tokenid >= vobsize, we remove these token ids")
                self.allowed_token_ids = [e for e in self.allowed_token_ids if e < vocab_size]
        return

    def has_constraint_setting(self) -> bool:
        return self.regular_constraint is not None or self.allowed_token_ids is not None


class InferReq:
    def __init__(
        self,
        shm_req: Req,
        req_idx: int,
        sampling_param: InferSamplingParams = None,
        multimodal_params=None,
    ) -> None:
        self.shm_req = shm_req
        self.sampling_param: InferSamplingParams = sampling_param
        self.multimodal_params = multimodal_params
        self.req_idx = req_idx
        self.shared_kv_node: TreeNode = None
        self.logprobs = []  # logprob of each token, using for beamsearch and diverse_backend
        self.cum_logprob = 0.0  # cumulative logprob of each token, using for beamsearch and diverse_backend

        self.cur_kv_len = 0
        self.cur_output_len = 0
        self.finish_status = FinishStatus()
        self.stop_sequences = self.sampling_param.shm_param.stop_sequences.to_list()

        # 标记是否完整完成初始化
        self.initialized = False
        return

    def init_all(self):
        """
        初始化 req 对象的 prompt ids 共享内存绑定以及radix cache的填充等
        """
        assert self.initialized is False

        # 区分是否是暂停恢复的请求, 暂停后恢复的请求不需要
        # link shm。
        if not hasattr(self.shm_req, "shm_prompt_ids"):
            self.shm_req.link_prompt_ids_shm_array()
            self.shm_req.link_logprobs_shm_array()
            if self.sampling_param.shm_param.input_penalty:
                self.out_token_id_count = collections.Counter(self.shm_req.get_prompt_ids())
            else:
                self.out_token_id_count = collections.defaultdict(int)

        # 如果是具有 prompt_cache 的使用特性则需要进行提前的填充和恢复操作。
        if g_infer_context.radix_cache is not None and self.get_cur_total_len() > 1:
            input_token_ids = self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]
            key = torch.tensor(input_token_ids, dtype=torch.int64, device="cpu")
            key = key[0 : len(key) - 1]  # 最后一个不需要，因为需要一个额外的token，让其在prefill的时候输出下一个token的值
            share_node, kv_len, value_tensor = g_infer_context.radix_cache.match_prefix(key, update_refs=True)
            if share_node is not None:
                self.shared_kv_node = share_node
                ready_cache_len = share_node.node_prefix_total_len
                g_infer_context.req_manager.req_to_token_indexs[self.req_idx, 0:ready_cache_len] = value_tensor
                self.cur_kv_len = int(ready_cache_len)  # 序列化问题, 该对象可能为numpy.int64

        self.shm_req.shm_cur_kv_len = self.cur_kv_len
        self.initialized = True
        return

    def get_output_len(self):
        return self.cur_output_len

    def get_cur_total_len(self):
        return self.shm_req.input_len + self.cur_output_len

    def get_input_token_ids(self):
        return self.shm_req.shm_prompt_ids.arr[0 : self.get_cur_total_len()]

    def set_next_gen_token_id(self, next_token_id: int, logprob: float):
        index = self.get_cur_total_len()
        self.shm_req.shm_prompt_ids.arr[index] = next_token_id
        self.shm_req.shm_logprobs.arr[index] = logprob
        return

    def get_last_gen_token(self):
        return self.shm_req.shm_prompt_ids.arr[self.shm_req.input_len + self.cur_output_len - 1]

    def update_finish_status(self, eos_ids):
        if self._stop_sequences_matched():
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif (
            self.cur_output_len > 0
            and self.get_last_gen_token() in eos_ids
            and self.sampling_param.shm_param.ignore_eos is False
        ):
            self.finish_status.set_status(FinishStatus.FINISHED_STOP)
        elif self.cur_output_len >= self.sampling_param.shm_param.max_new_tokens:
            self.finish_status.set_status(FinishStatus.FINISHED_LENGTH)
        return

    def _stop_sequences_matched(self):
        for stop_token_ids in self.stop_sequences:
            stop_len = len(stop_token_ids)
            output_len = self.cur_output_len
            if stop_len > 0:
                if output_len >= stop_len:
                    input_token_ids = self.shm_req.shm_prompt_ids.arr[
                        0 : (self.shm_req.input_len + self.cur_output_len)
                    ]
                    if all(input_token_ids[i] == stop_token_ids[i] for i in range(-1, -(stop_len + 1), -1)):
                        return True
        return False


class InferReqGroup:
    def __init__(
        self,
        group_req_id,
        best_of=1,
    ) -> None:
        self.best_of = best_of
        self.refs = best_of
        self.group_req_id = group_req_id
        self.prev_beamid = [0] * best_of  # Record which beam the current token came from.
        self.min_score = 1e9  # The min score of beam results
        self.req_group = []
        self.res = []  # Results already finished
        self.filter_reqs_id = []  # filtered reqs
        self.finish_status = False  # If beamsearch is done
        self.has_beam = False

    def get_req(self, index):
        return g_infer_context.requests_mapping[self.req_group[index]]

    def get_relative_index(self, index):
        return self.req_group[index] - self.group_req_id

    def add_req(self, req_id):
        self.req_group.append(req_id)

    def get_cumlogprobs(self):
        return [g_infer_context.requests_mapping[r_id].cum_logprob for r_id in self.req_group]

    def decrease_refs(self, req_id):
        self.refs -= 1
        self.filter_reqs_id.append(req_id)
        return self.refs == 0

    def update_filter(self):
        filter_reqs_id_set = set(self.filter_reqs_id)
        new_req_group = [req_id for req_id in self.req_group if req_id not in filter_reqs_id_set]
        self.req_group = new_req_group
        self.best_of = len(self.req_group)

    def add_res(self, output_ids, logprobs, cum_logprob, finish_status):
        score = cum_logprob / len(output_ids)
        if len(self.res) < self.best_of or score < self.min_score:
            self.res.append([score, output_ids, logprobs, finish_status])
            if len(self.res) > self.best_of:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _, _) in enumerate(self.res)])
                del self.res[sorted_scores[0][1]]
                self.min_score = sorted_scores[1][0]
            else:
                self.min_score = min(score, self.min_score)

    def beam_copy(self, req_manager, is_prefill):
        cache_req = {}
        cache_req_to_token = {}
        # record previous status
        for i, prev_ in enumerate(self.prev_beamid):
            prev_req = g_infer_context.requests_mapping[self.req_group[prev_]]
            if prev_ not in cache_req_to_token:
                cache_req[prev_] = copy.deepcopy(prev_req)
                prev_tokens = req_manager.req_to_token_indexs[prev_req.req_idx][: len(prev_req.input_token_ids)].clone()
                cache_req_to_token[prev_] = prev_tokens
            req = self.get_req(i)
            if not is_prefill or i == 0:
                req_manager.mem_manager.free(req_manager.req_to_token_indexs[req.req_idx][: len(req.input_token_ids)])

        # update the InferReq status and mem_manager status for cache sharing
        for i, req_id in enumerate(self.req_group):
            prev_ = self.prev_beamid[i]
            prev_req = cache_req[prev_]
            req = g_infer_context.requests_mapping[req_id]
            req.input_token_ids = copy.deepcopy(prev_req.input_token_ids)
            req.out_token_id_count = copy.deepcopy(req.out_token_id_count)
            req.logprobs = copy.deepcopy(req.logprobs)
            req.finish_status = FinishStatus.NO_FINISH
            req_manager.req_to_token_indexs[req.req_idx][: len(req.input_token_ids)] = cache_req_to_token[prev_]
            req_manager.mem_manager.add_refs(cache_req_to_token[prev_])

    def diverse_copy(self, req_manager, is_prefill):
        # record previous status
        prev_req = g_infer_context.requests_mapping[self.req_group[0]]
        if prev_req.shared_kv_node is not None:
            prefix_len = prev_req.shared_kv_node.node_prefix_total_len
        else:
            prefix_len = 0
        cache_token_id = req_manager.req_to_token_indexs[prev_req.req_idx][prefix_len : len(prev_req.input_token_ids)]
        # update the InferReq status and mem_manager status for cache sharing
        for req_id in self.req_group[1:]:
            req = g_infer_context.requests_mapping[req_id]
            req.finish_status = FinishStatus.NO_FINISH
            req_manager.req_to_token_indexs[req.req_idx][prefix_len : len(req.input_token_ids)] = cache_token_id
            assert len(req.input_token_ids) == len(prev_req.input_token_ids)
            req_manager.mem_manager.add_refs(cache_token_id)

    def update_finish_status(self, best_new_score):
        if len(self.res) < self.best_of:
            self.finish_status = False
        else:
            req_obj = g_infer_context.requests_mapping[self.req_group[0]]
            max_new_tokens = req_obj.sampling_param.max_new_tokens
            has_output_len = req_obj.cur_kv_len + 1 - req_obj.prompt_len  # 这个地方只能用 cur_kv_len 来计算，有beam流程函数存在依赖
            self.finish_status = best_new_score <= self.min_score or has_output_len >= max_new_tokens

        if self.finish_status:
            self.res = sorted(self.res, key=lambda i: i[0])
