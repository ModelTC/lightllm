import os
import math
import ctypes
import numpy as np
from .sampling_params import SamplingParams
from .out_token_circlequeue import CircularQueue
from .shm_array import ShmArray
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.utils.envs_utils import get_unique_server_name
from typing import List, Any, Union


class FinishStatus(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("status", ctypes.c_int)]

    NO_FINISH = 0
    FINISHED_STOP = 1
    FINISHED_LENGTH = 2

    def __init__(self, init_state=NO_FINISH):
        self.status = init_state

    def set_status(self, new_status):
        assert 0 <= new_status <= 2
        self.status = new_status

    def get_status(self):
        return self.status

    def is_finished(self):
        return self.FINISHED_STOP <= self.status <= self.FINISHED_LENGTH

    def get_finish_reason(self):
        if self.status == self.FINISHED_STOP:
            return "stop"
        elif self.status == self.FINISHED_LENGTH:
            return "length"
        return None


class PrefixTokenIdsStruct(ctypes.Structure):
    _pack_ = 4
    _fields_ = [("size", ctypes.c_int), ("data", ctypes.c_int64 * 10)]

    def __init__(self):
        self.size = 0

    def set_token_ids(self, ids: List[int]):
        self.size = len(ids)
        self.data[: len(ids)] = ids

    def get_token_ids(self):
        return list(self.data[: self.size])


class Req(ctypes.Structure):
    _pack_ = 4
    _fields_ = [
        ("index_in_shm_mem", ctypes.c_int),
        ("ref_count", ctypes.c_int),  # 个人不要操作这个计数  # 个人不要操作这个引用计数
        ("request_id", ctypes.c_int),  # 引用计数
        ("group_req_id", ctypes.c_int),
        ("input_len", ctypes.c_int),
        ("alloc_shm_numpy_len", ctypes.c_int),
        ("shm_infer_released", ctypes.c_bool),  # 推理进程用于标记请求对象已经被推理进程释放，router进程得到信息后亦可释放shm req对象
        ("shm_cur_kv_len", ctypes.c_int),  # 推理进程记录自己当前占用kv 显存长度
        ("shm_cur_output_len", ctypes.c_int),  # 推理进程记录自己输出长度的计数
        # candetoken_out_len 推理进程修改这个数据，让detokenization进程知道需要detoken的长度，
        # 虽然某种程度上 cur_output_len 也有同样的功能，但是为了避免多进程访问导致的问题，添加
        # candetoken_out_len 变量单独传输这个信息。
        ("candetoken_out_len", ctypes.c_int),
        ("prompt_cache_len", ctypes.c_int),  # 用于记录prompt cache 的命中长度，用于统计
        ("is_paused", ctypes.c_bool),  # 标记一个Req因为显存资源管理的原因被临时暂停了。
        ("finish_status", FinishStatus),
        ("is_aborted", ctypes.c_bool),
        # 这个标记变量是router进程读取到is_aborted信息后，将这个router_aborted 变量标记为True，因为推理进程
        # 直接读取 is_aborted 变量可能会存在异步问题，但是router的执行线程和推理进程之间是线性运行的，所以router
        # 进程写入的router_aborted信息，所有推理进程可以保证同时读取到的是正确信息，不会出现异步问题。
        ("router_aborted", ctypes.c_bool),
        # 当FinishStatus 是正常结束状态时，finish_token_index 用于标识结束的
        # token 的index位置
        ("finish_token_index", ctypes.c_int),
        ("out_tokens_queue", CircularQueue),
        ("sample_params", SamplingParams),
        ("chunked_prefill_size", ctypes.c_int),  # 只有chunked prefill模式才使用的参数
        ("remaining_prefill_size", ctypes.c_int),  # 只有chunked prefill模式才使用的参数
        ("prefix_token_ids", PrefixTokenIdsStruct),  # 只有 token_headling 模式使用的参数
        # can_released_mark的作用是：
        # 只有整个流程中的最后一个处理模块，一般是 detokenization 进程，标记这个参数为True后，主管理进程才能真
        # 的释放请求对像。
        ("can_released_mark", ctypes.c_bool),
        # reward_model 使用的变量
        ("reward_score", ctypes.c_float),
    ]

    def init(
        self,
        request_id: int,
        prompt_ids: List[int],
        sample_param: Union[dict, SamplingParams],
        tokenizer: Any,
        chunked_prefill_size: int = 0,
    ):
        # 只是为了有更好的编码辅助类型提示
        self.index_in_shm_mem: int = self.index_in_shm_mem
        self.ref_count: int = self.ref_count

        self.request_id = request_id
        self.group_req_id = convert_sub_id_to_group_id(request_id)
        self.is_paused = False
        self.finish_status = FinishStatus()
        self.is_aborted = False
        self.router_aborted = False
        self.shm_infer_released = False
        self.shm_cur_kv_len = 0
        self.shm_cur_output_len = 0
        self.candetoken_out_len = 0
        self.prompt_cache_len = 0
        self.finish_token_index = -1
        self.can_released_mark = False
        self.reward_score = math.nan
        if isinstance(sample_param, SamplingParams):
            self.sample_params = sample_param
        else:
            self.sample_params = SamplingParams()
            self.sample_params.init(tokenizer=tokenizer, **sample_param)
        self.chunked_prefill_size = chunked_prefill_size
        self.remaining_prefill_size = len(prompt_ids) if chunked_prefill_size > 0 else 0
        self.prefix_token_ids = PrefixTokenIdsStruct()

        self.out_tokens_queue = CircularQueue()
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
        self.shm_prompt_ids = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.int64)
        self.shm_prompt_ids.create_shm()
        return

    def link_prompt_ids_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_prompts_{self.index_in_shm_mem}"
        self.shm_prompt_ids = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.int64)
        self.shm_prompt_ids.link_shm()
        return

    def create_logprobs_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_logprobs_{self.index_in_shm_mem}"
        self.shm_logprobs = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.float32)
        self.shm_logprobs.create_shm()
        return

    def link_logprobs_shm_array(self):
        service_uni_name = get_unique_server_name()
        name = f"{service_uni_name}_shm_logprobs_{self.index_in_shm_mem}"
        self.shm_logprobs = ShmArray(name, (self.alloc_shm_numpy_len,), dtype=np.float32)
        self.shm_logprobs.link_shm()
        return

    def get_prompt_ids(self):
        return self.shm_prompt_ids.arr[: self.input_len].tolist()

    def to_router_rpc_obj(self):
        if hasattr(self, "multimodal_params"):
            return (
                self.request_id,
                self.index_in_shm_mem,
                self.multimodal_params,
                self.sample_params.suggested_dp_index,
            )
        else:
            return (self.request_id, self.index_in_shm_mem, None, self.sample_params.suggested_dp_index)

    def can_release(self):
        # 只有管理节点有一个引用
        ref_count_ok = self.ref_count == 1
        can_released_mark = self.can_released_mark

        if self.is_aborted and can_released_mark and ref_count_ok:
            return True

        if self.finish_status.is_finished() and can_released_mark and ref_count_ok and self.out_tokens_queue.is_empty():
            return True

        return False

    def get_used_tokens(self):
        return max(0, self.shm_cur_kv_len)

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        raise NotImplementedError("Subclasses should implement this method")

    def get_decode_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_first_router_need_tokens(self):
        raise NotImplementedError("Subclasses should implement this method")

    def get_all_prompt_metadata(self):
        """
        return_all_prompt_logprobs mode use to return all logprobs cacul ppl
        """
        metadata = {}
        cur_ids = self.shm_prompt_ids.arr[0 : self.input_len]
        all_prompts = []
        for index in range(len(cur_ids) - 1):
            tmp_dict = {int(cur_ids[index + 1]): float(self.shm_logprobs.arr[index + 1])}
            all_prompts.append([int(cur_ids[index]), tmp_dict])

        metadata["prompt_logprobs"] = all_prompts
        metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
        return metadata

    def update_remaining_prefill_size(self):
        self.remaining_prefill_size = 0
        return


# 由于目前加入了很多异步调度的方法，为了缓解异步调度带来的很多
# 估计不准确的问题，通过加长输出的长度，进行偏向保守一些的调度
# 理论上不会多估计太多的 token 占用量, 同时得到较高的token显存
# 使用率
ADDED_OUTPUT_LEN = 6


class NormalReq(Req):
    _pack_ = 4

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        has_out_len = self.shm_cur_output_len
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

        a_len = max(self.input_len + has_out_len + 1, self.shm_cur_kv_len + 1)
        b_len = max(0, cur_max_new_token_len - has_out_len - 1) + ADDED_OUTPUT_LEN

        return (a_len, b_len)

    def get_decode_need_tokens(self):

        return 1

    def get_first_router_need_tokens(self):

        return self.input_len + self.shm_cur_output_len


class TokenHealingReq(NormalReq):
    _pack_ = 4

    def post_init(
        self,
    ):
        for prefix_token_num in range(2, -1, -1):
            if self.input_len > prefix_token_num:
                self.input_len -= prefix_token_num
                self.prefix_token_ids.set_token_ids(
                    self.shm_prompt_ids.arr[self.input_len : (self.input_len + prefix_token_num)]
                )
                break

        # 因为原始的输出token数量，会被中间的前缀补全占用decode次数，
        # 所以默认多添加一些decode步数, token healing mode 下，由于
        # 估计的生成token数据对应的生存周期可能会不准确,所以为了缓解调
        # 度带来的显存估计问题，对于生成token的长度 + 6来缓解可能的估计
        # 错误问题。
        self.sample_params.max_new_tokens = self.sample_params.max_new_tokens + self.prefix_token_ids.size + 6
        return


class ChunkedPrefillReq(Req):
    _pack_ = 4

    def get_tuple_tokens(self, is_busy, router_max_new_token_len):
        has_out_len = self.shm_cur_output_len
        if self.sample_params.ignore_eos:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        elif is_busy:
            cur_max_new_token_len = self.sample_params.max_new_tokens
        else:
            cur_max_new_token_len = min(
                self.sample_params.max_new_tokens, max(int(1.1 * has_out_len), router_max_new_token_len)
            )

        a_len = max(self.input_len + has_out_len + 1, self.shm_cur_kv_len + 1)
        b_len = (
            (self.input_len + has_out_len - self.shm_cur_kv_len + self.splitfuse_block_size - 1)
            // self.splitfuse_block_size
            + cur_max_new_token_len
            - has_out_len
            - 1
        )
        b_len = max(0, b_len) + ADDED_OUTPUT_LEN

        return (a_len, b_len)

    def get_decode_need_tokens(self):
        """
        splitfuse 调度模式的实现
        """
        return min(self.input_len + self.shm_cur_output_len - self.shm_cur_kv_len, self.splitfuse_block_size)

    def get_first_router_need_tokens(self):

        return min(self.input_len + self.shm_cur_output_len, self.splitfuse_block_size)
