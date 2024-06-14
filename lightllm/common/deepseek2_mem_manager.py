import torch

from .mem_manager import MemoryManager


class Deepseek2MemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, key_head_dim, value_head_dim, layer_num, always_copy=True):
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        super().__init__(size, dtype, head_num, -1, layer_num, always_copy=True)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.k_buffer = [
            torch.empty((size, head_num, self.key_head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)
        ]
        self.v_buffer = [
            torch.empty((size, head_num, self.value_head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)
        ]
