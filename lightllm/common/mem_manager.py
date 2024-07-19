import re
import os
import torch
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt

logger = init_logger(__name__)


class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, device_type="cuda"):
        self.size = size
        self.dtype = dtype
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        assert device_type in ["cuda", "cpu"]
        self.device_type = device_type

        # mem_state 修改为使用计数方式，方便后期实现token共享机制，实现beam search 等
        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size

        # 用共享内存进行共享，router 模块读取进行精确的调度估计, nccl port 作为一个单机中单实列的标记。防止冲突。
        from torch.distributed.distributed_c10d import _default_pg_init_method

        nccl_port = re.search(r":(\d+)$", _default_pg_init_method).group(1)
        assert nccl_port is not None
        logger.info(f"mem manger get nccl port: {str(nccl_port)}")

        self.shared_can_use_token_num = SharedInt(f"{str(nccl_port)}_{device_type}_mem_manger_can_use_token_num")

        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.device_type == "cpu":
            self.pin_memory = True
        else:
            self.pin_memory = False

        self._init_buffers(size, dtype, head_num, head_dim, layer_num)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.empty(
                (size, 2 * head_num, head_dim), dtype=dtype, pin_memory=self.pin_memory, device=self.device_type
            )
            for _ in range(layer_num)
        ]

    def build_cpu_cache_mem_manger(self, cpu_cache_total_token_num):
        """
        用同样的参数，除了token num 不一样的，构建一个device_type 为 cpu的mem manager 对象
        """
        if cpu_cache_total_token_num == 0:
            return None
        return type(self)(
            cpu_cache_total_token_num,
            self.dtype,
            self.head_num,
            self.head_dim,
            self.layer_num,
            always_copy=self.always_copy,
            device_type="cpu",
        )

    def copy_to_mem_manager(self, source_index: torch.Tensor, mem_manager, dest_index: torch.Tensor):
        """
        将当前mem manager 中的数据拷贝到其他mem_manager中
        所有子类需要继承实现该函数, 实现在不同 mem_manager 中的数据移动
        """
        from gpu_cpu_swap import swap_data_by_index

        for layer_index in range(len(self.kv_buffer)):
            # to do, 高效率的数据传输需要精确的设置grid_num 和 wrap_num
            grid_num = min(len(source_index), 200)
            wrap_num = 1
            swap_data_by_index(
                mem_manager.kv_buffer[layer_index].view(self.size, -1),
                dest_index.cuda(),
                self.kv_buffer[layer_index].view(self.size, -1),
                source_index.cuda(),
                grid_num,
                wrap_num,
            )
        return

    def _free_buffers(self):
        self.kv_buffer = None

    @torch.no_grad()
    def alloc(self, need_size):
        if need_size > self.can_use_mem_size:
            logger.warn(
                f"{self.device_type} warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None
        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        select_index = can_use_index[0:need_size]
        self.add_refs(select_index)
        return select_index

    @torch.no_grad()
    def alloc_contiguous(self, need_size):
        if self.always_copy:
            return None
        if need_size > self.can_use_mem_size:
            logger.warn(
                f"{self.device_type} warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_index = torch.nonzero(self.mem_state == 0).view(-1)
        can_use_index_size = len(can_use_index)
        can_use_index = can_use_index[0 : can_use_index_size - need_size + 1][
            (can_use_index[need_size - 1 :] - can_use_index[0 : can_use_index_size - need_size + 1]) == need_size - 1
        ]
        if can_use_index.shape[0] == 0:
            # logger.warn(f'warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}')
            return None
        start = can_use_index[0].item()
        end = start + need_size
        select_index = self.indexes[start:end]
        self.add_refs(select_index)
        return select_index, start, end

    @torch.no_grad()
    def free(self, free_index):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """
        free_index = free_index.long()
        self.decrease_refs(free_index)
        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"{self.device_type} freed all gpu mem size {self.can_use_mem_size}")
        return

    @torch.no_grad()
    def add_refs(self, token_index: torch.Tensor):
        state = self.mem_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state[token_index] += 1
        return

    @torch.no_grad()
    def decrease_refs(self, token_index: torch.Tensor):
        token_index, counts = token_index.unique(return_counts=True)
        self.mem_state[token_index] -= counts
        state = self.mem_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        return

    @torch.no_grad()
    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state[:] = 0

    @torch.no_grad()
    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num

        self.mem_state = torch.zeros((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        return
