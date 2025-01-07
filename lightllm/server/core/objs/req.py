import os
import ctypes
import numpy as np
from .sampling_params import SamplingParams
from .shm_array import ShmArray
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.utils.envs_utils import get_unique_server_name
from typing import List, Any


class ReqStatus(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int)]

    WAIT_IN_QUEUE = 0
    RUNNING = 1
    PAUSED_AND_OFFLOAD = 2
    RERUNNING_FROM_OFFLOAD = 3

    def __init__(self):
        self.status = self.WAIT_IN_QUEUE

    def set_status(self, new_status):
        assert 0 <= new_status <= 3
        self.status = new_status

    def get_status(self):
        return self.status

    def is_waiting(self):
        return self.status == self.WAIT_IN_QUEUE

    def is_running(self):
        return self.status == self.RUNNING

    def is_paused_and_offload(self):
        return self.status == self.PAUSED_AND_OFFLOAD

    def is_rerunning(self):
        return self.status == self.RERUNNING_FROM_OFFLOAD


class FinishStatus(ctypes.Structure):
    _fields_ = [("status", ctypes.c_int)]

    NO_FINISH = 0
    FINISHED_STOP = 1
    FINISHED_LENGTH = 2
    FINISHED_ABORT = 3

    def __init__(self):
        self.status = self.NO_FINISH

    def set_status(self, new_status):
        assert 0 <= new_status <= 3
        self.status = new_status

    def get_status(self):
        return self.status

    def is_finished(self):
        return self.FINISHED_STOP <= self.status <= self.FINISHED_ABORT

    def is_aborted(self):
        return self.status == self.FINISHED_ABORT

    def get_finish_reason(self):
        if self.status == self.FINISHED_STOP:
            return "stop"
        elif self.status == self.FINISHED_LENGTH:
            return "length"
        elif self.status == self.FINISHED_ABORT:
            return "abort"
        return None


class PrefixTokenIdsStruct(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int), ("data", ctypes.c_int64 * 10)]

    def __init__(self):
        self.size = 0


class Req(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("index_in_shm_mem", ctypes.c_int),
        ("ref_count", ctypes.c_int),  # 个人不要操作这个计数  # 个人不要操作这个引用计数
        ("request_id", ctypes.c_int),  # 引用计数
        ("group_req_id", ctypes.c_int),
        ("input_len", ctypes.c_int),
        ("alloc_shm_numpy_len", ctypes.c_int),
        ("cur_kv_len", ctypes.c_int),
        ("cur_output_len", ctypes.c_int),
        ("prompt_cache_len", ctypes.c_int),
        ("req_status", ReqStatus),
        ("finish_status", FinishStatus),
        ("sample_params", SamplingParams),
        ("splitfuse_block_size", ctypes.c_int),  # 只有splitfuse模式才使用的参数
        ("prefix_token_ids", PrefixTokenIdsStruct),  # 只有 token_headling 模式使用的参数
    ]

    def init(self, request_id: int, prompt_ids: List[int], sample_param: dict, tokenizer: Any, splitfuse_block_size=0):
        self.request_id = request_id
        self.group_req_id = convert_sub_id_to_group_id(request_id)
        self.req_status = ReqStatus()
        self.finish_status = FinishStatus()
        self.cur_kv_len = 0
        self.cur_output_len = 0
        self.prompt_cache_len = 0
        self.sample_params = SamplingParams()
        self.sample_params.init(tokenizer=tokenizer, **sample_param)
        self.splitfuse_block_size = splitfuse_block_size
        self.prefix_token_ids = PrefixTokenIdsStruct()

        self.input_len = len(prompt_ids)
        self.alloc_shm_numpy_len = self.input_len + self.sample_params.max_new_tokens + 1024  # + 1024 for safe
        self.create_logprobs_shm_array()
        self.create_prompt_ids_shm_array()

        self.shm_prompt_ids.arr[0 : len(prompt_ids)] = prompt_ids

        self.post_init()

    def post_init(self):
        # 子类继承进行一些额外的初始化操作
        pass

    def create_prompt_ids_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_prompts_{self.index_in_shm_mem}"
        assert not hasattr(self, "shm_prompt_ids")
        self.shm_prompt_ids = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.int64)
        self.shm_prompt_ids.create_shm()
        return

    def link_prompt_ids_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_prompts_{self.index_in_shm_mem}"
        assert not hasattr(self, "shm_prompt_ids")
        self.shm_prompt_ids = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.int64)
        self.shm_prompt_ids.link_shm()
        return

    def create_logprobs_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_logprobs_{self.index_in_shm_mem}"
        assert not hasattr(self, "shm_logprobs")
        self.shm_logprobs = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.float32)
        self.shm_logprobs.create_shm()
        return

    def link_logprobs_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_logprobs_{self.index_in_shm_mem}"
        assert not hasattr(self, "shm_logprobs")
        self.shm_logprobs = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.float32)
        self.shm_logprobs.link_shm()
        return

    def get_used_tokens(self):
        return max(0, self.cur_kv_len)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise NotImplementedError("Subclasses should implement this method")

    def get_decode_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_first_router_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def __del__(self):
        pass


class NormalReq(Req):
    _pack_ = 4

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        has_out_len = self.cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        elif is_busy:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        else:
            # 用当前输出长度的 1.1 倍作为预估输出长度的另一个参考量，用于更新估计的最大输出长度量
            # 后续会更新为更合理的统计条件概率估计方式 to do
            cur_max_new_token_len = min(
                self.sample_params.max_new_tokens, max(int(1.1 * has_out_len), router_max_new_token_len)
            )

        if self.req_status.is_running():
            return (self.input_len + has_out_len, max(0, cur_max_new_token_len - has_out_len - 1))
        elif self.req_status.is_waiting():
            return (self.input_len + 1, max(0, cur_max_new_token_len - 1 - 1))
        elif self.req_status.is_paused_and_offload():
            return (self.input_len + has_out_len + 1, max(0, cur_max_new_token_len - has_out_len - 1 - 1))
        else:
            raise ValueError("Invalid request status")

    def get_decode_need_tokens(self):
        if self.req_status.is_running():
            return 1
        else:
            raise ValueError("Invalid request status")

    def get_first_router_need_tokens(self):
        if self.req_status.is_waiting():
            return self.input_len
        elif self.req_status.is_paused_and_offload():
            return self.input_len + self.cur_output_len
        else:
            raise ValueError("Invalid request status")


class TokenHealingReq(NormalReq):
    _pack_ = 4

    def post_init(
        self,
    ):
        for prefix_token_num in range(2, -1, -1):
            if self.input_len > prefix_token_num:
                self.input_len -= prefix_token_num
                for get_token_id_idx in range(prefix_token_num):
                    self.prefix_token_ids.data[get_token_id_idx] = self.prompt_ids[self.input_len + get_token_id_idx]
                self.prefix_token_ids.size = prefix_token_num
                break
        # 因为原始的输出token数量，会被中间的前缀补全占用decode次数，
        # 所以默认多添加一些decode步数
        # token healing mode 下，由于估计的生成token数据对应的生存周期可能会不准确
        # 所以为了缓解调度带来的显存估计问题，对于生成token的长度 + 3来缓解可能的估计
        # 错误问题。
        self.sample_params.max_new_tokens = self.sample_params.max_new_tokens + self.prefix_token_ids.size + 6
        return


class SplitFuseReq(Req):
    _pack_ = 4

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        has_out_len = self.cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        elif is_busy:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        else:
            cur_max_new_token_len = min(
                self.sample_params.max_new_tokens, max(int(1.1 * has_out_len), router_max_new_token_len)
            )

        if self.req_status.is_running():
            return (
                self.input_len + has_out_len,
                max(
                    0,
                    (self.input_len + has_out_len - self.cur_kv_len + self.splitfuse_block_size - 1)
                    // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - has_out_len
                    - 1,
                ),
            )
        elif self.req_status.is_waiting():
            return (
                self.input_len,
                max(
                    0,
                    (self.input_len + self.splitfuse_block_size - 1) // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - 1,
                ),
            )
        elif self.req_status.is_paused_and_offload():
            return (
                self.input_len + has_out_len,
                max(
                    0,
                    (self.input_len + has_out_len + self.splitfuse_block_size - 1) // self.splitfuse_block_size
                    + cur_max_new_token_len
                    - has_out_len
                    - 1,
                ),
            )
        else:
            raise ValueError("Invalid request status")

    def get_decode_need_tokens(self):
        """
        splitfuse 调度模式的实现
        """
        if self.req_status.is_running():
            return min(self.input_len + self.cur_output_len - self.cur_kv_len, self.splitfuse_block_size)
        else:
            raise ValueError("Invalid request status")

    def get_first_router_need_tokens(self):
        if self.req_status.is_waiting():
            return min(self.input_len, self.splitfuse_block_size)
        elif self.req_status.is_paused_and_offload():
            return min(self.input_len + self.cur_output_len, self.splitfuse_block_size)
        else:
            raise ValueError("Invalid request status")
