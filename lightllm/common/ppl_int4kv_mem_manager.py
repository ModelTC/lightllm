import torch

from .mem_manager import MemoryManager


class PPLINT4KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=True):
        self.kv_dtype = torch.int8
        self.group_quant_size = 8
        super().__init__(size, dtype, head_num, head_dim, layer_num, always_copy=True)

    def get_cell_size(self):
        return self.head_num * self.head_dim // 2 * self.layer_num * 2 * torch._utils._element_size(
            self.kv_dtype
        ) + self.head_num * self.head_dim // self.group_quant_size * self.layer_num * torch._utils._element_size(
            self.dtype
        )

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.empty((size, 2 * head_num, head_dim // 2), dtype=torch.int8, device="cuda") for _ in range(layer_num)
        ]
        self.scale_buffer = [
            torch.empty((size, 2 * head_num, head_dim // self.group_quant_size), dtype=dtype, device="cuda")
            for _ in range(layer_num)
        ]

    def _free_buffers(self):
        self.kv_buffer = None
        self.scale_buffer = None
