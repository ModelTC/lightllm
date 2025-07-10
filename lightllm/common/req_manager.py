import torch
import collections
from lightllm.utils.log_utils import init_logger
from .mem_manager import MemoryManager
from typing import List, Optional
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import token_id_counter
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import update_req_to_token_id_counter
from lightllm.utils.envs_utils import enable_env_vars, get_env_start_args
from lightllm.utils.config_utils import get_vocab_size

logger = init_logger(__name__)


class _ReqNode:
    def __init__(self, index):
        self.index = index
        self.next: "_ReqNode" = None


class _ReqLinkedList:
    def __init__(self, max_request_num):
        self.nodes = [_ReqNode(i) for i in range(max_request_num)]
        self.marks = [0 for _ in range(max_request_num)]
        self.root_node = _ReqNode(-1)
        for i in range(0, max_request_num - 1):
            self.nodes[i].next = self.nodes[i + 1]
        self.root_node.next = self.nodes[0]
        self.can_alloc_size = max_request_num
        return

    def alloc(self):
        if self.root_node.next is None:
            logger.warning("alloc req index fail")
            return None
        get_node = self.root_node.next
        self.root_node.next = self.root_node.next.next
        assert self.marks[get_node.index] == 0
        self.marks[get_node.index] = 1
        self.can_alloc_size -= 1
        return get_node.index

    def free(self, index):
        assert self.marks[index] == 1
        node = self.nodes[index]
        node.next = self.root_node.next
        self.root_node.next = node
        self.marks[index] = 0
        self.can_alloc_size += 1
        return

    def is_all_free(self):
        return self.can_alloc_size == len(self.marks)


class ReqManager:
    def __init__(self, max_request_num, max_sequence_length, mem_manager: MemoryManager):
        # 这里对最大请求数量的管理在默认上多申请了一个，主要是 index 为 max_request_num 代表
        # 的这个请求管理 id， 主要是为了兼容 DP 运行模式下，让各个 DP 能 padding 到 DP 中最大
        # 的那个batch size 进行运行，所有 padding 的请求都会使用预留的这个请求管理 id 进行处理
        # 这样让 DP 的实现更为简化一些。
        self.req_list = _ReqLinkedList(max_request_num)
        self.req_to_token_indexs = torch.zeros(
            (max_request_num + 1, max_sequence_length), dtype=torch.int32, device="cuda"
        )
        self.mem_manager = mem_manager
        self.req_sampling_params_manager = ReqSamplingParamsManager(max_request_num)
        self.max_request_num = max_request_num
        self.HOLD_REQUEST_ID = max_request_num

    def alloc(self):
        return self.req_list.alloc()

    def free(self, free_req_indexes: List[int], free_token_index):
        for req_index in free_req_indexes:
            self.req_list.free(req_index)

        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        self.mem_manager.free(free_token_index)

    def free_req(self, free_req_index: int):
        self.req_list.free(free_req_index)
        if self.req_list.is_all_free():
            logger.debug(f"freed all request size {self.req_list.can_alloc_size}")
        return

    def free_token(self, free_token_index):
        self.mem_manager.free(free_token_index)
        return

    def free_all(self):
        self.req_list = _ReqLinkedList(self.max_request_num)
        return


class ReqSamplingParamsManager:
    """
    ReqSamplingParamsManager 将输出采样参数中，确定比较固定的部分，纳入到 gpu buffer中进行管理，这样可以更快捷的
    利用 triton kernel 进行处理，对于那些比较动态(部分处理模式下会动态的修改某些后处理参数)，或者存在特殊处理的后处理参数，
    则保留从 InferSamplingParams 中进行动态读取和动态组batch， 具体使用可以参考
    lightllm/server/router/model_infer/mode_backend/generic_post_process.py 文件中的使用方式。
    """

    def __init__(self, max_request_num):
        self.enable_gpu_buffer_for_out_token_id_counter: bool = enable_env_vars(
            "LIGHTLLM_ENABLE_GPU_BUFFER_FOR_OUT_TOKEN_ID_COUNTER"
        )
        self.vocab_size = get_vocab_size(get_env_start_args().model_dir)
        self.req_to_next_token_id = torch.zeros(max_request_num + 1, dtype=torch.int64, device="cuda")
        self.req_to_presence_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_frequency_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_repetition_penalty = torch.zeros(max_request_num + 1, dtype=torch.float32, device="cuda")
        self.req_to_exponential_decay_length_penalty = torch.zeros(
            max_request_num + 1, dtype=torch.float32, device="cuda"
        )

        if self.enable_gpu_buffer_for_out_token_id_counter:
            self.req_to_out_token_id_counter = torch.zeros(
                (max_request_num + 1, self.vocab_size), dtype=torch.int32, device="cuda"
            )

        # 用于缓存用的 pin_memory tensor，减少频繁初始化可以提升性能，因为microbatch overlap折叠的原因
        # 需要两套 buffer 来生成缓存
        self._p_token_ids_buffer: List[torch.Tensor] = [
            torch.zeros((0,), dtype=torch.int32, device="cpu", pin_memory=True) for _ in range(2)
        ]
        self._p_token_counts_buffer: List[torch.Tensor] = [
            torch.zeros((0,), dtype=torch.int32, device="cpu", pin_memory=True) for _ in range(2)
        ]
        self._p_cumsum_seq_len_buffer: List[torch.Tensor] = [
            torch.zeros((0,), dtype=torch.int32, device="cpu", pin_memory=True) for _ in range(2)
        ]

    def init_req_sampling_params(self, req):
        # fix cycle loop import
        from lightllm.server.router.model_infer.infer_batch import InferReq

        req: InferReq = req

        shm_param = req.sampling_param.shm_param
        self.req_to_next_token_id[req.req_idx].fill_(req.get_last_gen_token())
        self.req_to_presence_penalty[req.req_idx].fill_(shm_param.presence_penalty)
        self.req_to_frequency_penalty[req.req_idx].fill_(shm_param.frequency_penalty)
        self.req_to_repetition_penalty[req.req_idx].fill_(shm_param.repetition_penalty)
        exponential_decay_length_penalty = shm_param.exponential_decay_length_penalty.to_tuple()
        self.req_to_exponential_decay_length_penalty[req.req_idx].fill_(exponential_decay_length_penalty[1])
        # 提前标记当前请求是否需要统计输出token的计数，因为这个统计可能会导致一些特定场景下后处理效率的下降
        # 所以提前标记不需要进行后处理统计的场景。
        req.need_out_token_id_statistics = not (
            shm_param.presence_penalty == 0.0
            and shm_param.frequency_penalty == 0.0
            and shm_param.repetition_penalty == 1.0
        )

        if not self.enable_gpu_buffer_for_out_token_id_counter:
            if req.sampling_param.shm_param.input_penalty and req.need_out_token_id_statistics:
                req.out_token_id_count = collections.Counter(req.shm_req.get_prompt_ids())
            else:
                req.out_token_id_count = collections.defaultdict(int)
        else:
            self.req_to_out_token_id_counter[req.req_idx].fill_(0)
            if req.sampling_param.shm_param.input_penalty and req.need_out_token_id_statistics:
                prompt_ids = torch.from_numpy(req.shm_req.get_prompt_ids()).pin_memory().cuda(non_blocking=True)
                token_id_counter(
                    prompt_ids=prompt_ids, out_token_id_counter=self.req_to_out_token_id_counter[req.req_idx]
                )

        return

    def update_reqs_token_counter(
        self, req_objs: List, next_token_ids: List[int], accept_mark: Optional[List[List[bool]]] = None
    ):
        from lightllm.server.router.model_infer.infer_batch import InferReq

        req_objs: List[InferReq] = req_objs

        if not self.enable_gpu_buffer_for_out_token_id_counter:
            for req_obj, next_token_id in zip(req_objs, next_token_ids):
                if req_obj.need_out_token_id_statistics and req_obj.cur_output_len > 0:
                    req_obj.out_token_id_count[next_token_id] += 1
        else:
            b_req_idx = torch.tensor(
                [req.req_idx for req in req_objs], dtype=torch.int32, device="cpu", pin_memory=True
            ).cuda(non_blocking=True)
            next_token_ids = (
                torch.tensor(next_token_ids, dtype=torch.int32, device="cpu").pin_memory().cuda(non_blocking=True)
            )
            update_req_to_token_id_counter(
                b_req_idx=b_req_idx,
                next_token_ids=next_token_ids,
                req_to_out_token_id_counter=self.req_to_out_token_id_counter,
            )
        return

    def gen_cpu_out_token_counter_sampling_params(self, req_objs: List, batch_index: int = 0):
        from lightllm.server.router.model_infer.infer_batch import InferReq

        req_objs: List[InferReq] = req_objs
        assert not self.enable_gpu_buffer_for_out_token_id_counter

        p_token_ids: List[int] = []
        p_token_counts: List[int] = []
        p_cumsum_seq_len: List[int] = [
            0,
        ]
        cum_sum_len = 0
        for i, req_obj in enumerate(req_objs):
            id_to_count = req_obj.out_token_id_count
            p_token_ids.extend(list(id_to_count.keys()))
            p_token_counts.extend(list(id_to_count.values()))
            cum_sum_len += len(id_to_count)
            p_cumsum_seq_len.append(cum_sum_len)

        p_token_ids = self._alloc_gpu_buffer_tensor_and_init(
            p_token_ids, self._p_token_ids_buffer, batch_index=batch_index
        )
        p_token_counts = self._alloc_gpu_buffer_tensor_and_init(
            p_token_counts, self._p_token_counts_buffer, batch_index=batch_index
        )
        p_cumsum_seq_len = self._alloc_gpu_buffer_tensor_and_init(
            p_cumsum_seq_len, self._p_cumsum_seq_len_buffer, batch_index=batch_index
        )
        return p_token_ids, p_token_counts, p_cumsum_seq_len

    def _alloc_gpu_buffer_tensor_and_init(self, in_data, buffer_list: List[torch.Tensor], batch_index: int = 0):
        size = len(in_data)

        if size > buffer_list[batch_index].shape[0]:
            buffer_list[batch_index] = torch.empty((size,), dtype=torch.int32, device="cpu", pin_memory=True)

        current_buffer = buffer_list[batch_index]
        current_buffer.numpy()[0:size] = in_data
        return current_buffer[0:size].cuda(non_blocking=True)
