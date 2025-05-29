import torch

from .mem_manager import MemoryManager


class FP8KVMemoryManager(MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        # scale被追加到kv_buffer末尾, 因此加4, dtype统一改成uint8
        super().__init__(size, torch.uint8, head_num, head_dim + 4, layer_num, always_copy, mem_fraction)
