import torch

from .mem_manager import MemoryManager


class INT8KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=True, device_type="cuda"):
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy=True, device_type=device_type)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.empty(
                (size, 2 * head_num, head_dim), dtype=torch.int8, pin_memory=self.pin_memory, device=self.device_type
            )
            for _ in range(layer_num)
        ]
        self.scale_buffer = [
            torch.empty((size, 2 * head_num, 1), dtype=dtype, pin_memory=self.pin_memory, device=self.device_type)
            for _ in range(layer_num)
        ]

    def _free_buffers(self):
        self.kv_buffer = None
        self.scale_buffer = None
