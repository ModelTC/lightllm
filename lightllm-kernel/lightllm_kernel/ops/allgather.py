import torch
from typing import Optional, List, Tuple
from . import _C


def all_gather(
    _fa: int, inp: torch.Tensor, out: torch.Tensor, _reg_buffer: int, reg_buffer_sz_bytes: int
) -> torch.Tensor:
    return _C.all_gather(_fa, inp, out, _reg_buffer, reg_buffer_sz_bytes)


def init_custom_gather_ar(fake_ipc_ptrs: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool) -> int:
    return _C.init_custom_gather_ar(fake_ipc_ptrs, rank_data, rank, full_nvlink)


def allgather_dispose(_fa: int) -> None:
    _C.allgather_dispose(_fa)


def allgather_register_buffer(_fa: int, fake_ipc_ptrs: List[int]) -> None:
    _C.allgather_register_buffer(_fa, fake_ipc_ptrs)


def allgather_get_graph_buffer_ipc_meta(_fa: int) -> Tuple[List[int], List[int]]:
    return _C.allgather_get_graph_buffer_ipc_meta(_fa)


def allgather_register_graph_buffers(_fa: int, handles: List[List[int]], offsets: List[List[int]]) -> None:
    _C.allgather_register_graph_buffers(_fa, handles, offsets)
