import torch

from .mem_manager import MemoryManager


class Deepseek2MemoryManager(MemoryManager):
    def __init__(self, size, dtype, kv_lora_rank, qk_rope_head_dim, layer_num, always_copy=True):
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        super().__init__(size, dtype, 1, -1, layer_num, always_copy=True)

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        self.kv_buffer = [
            torch.empty((size, 1, self.kv_lora_rank + self.qk_rope_head_dim), dtype=dtype, device="cuda") for _ in range(layer_num)
        ]
