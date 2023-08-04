import numpy as np
import torch
from typing import Union, List
from . import np_from_tensor


def init_bloc(b_loc: torch.Tensor,
              b_seq_len: Union[np.ndarray, List[int], torch.Tensor],
              max_len_in_batch: int,
              alloc_mem_index: torch.Tensor,
              ):
    start_index = 0
    if isinstance(b_seq_len, (np.ndarray, list)):
        b_seq_len_lst = b_seq_len
    elif isinstance(b_seq_len, torch.Tensor):
        b_seq_len_lst = np_from_tensor(b_seq_len)
    for i, cur_seq_len in enumerate(b_seq_len_lst):
        b_loc[i, max_len_in_batch - cur_seq_len:max_len_in_batch] = alloc_mem_index[start_index:start_index + cur_seq_len]
        start_index += cur_seq_len
    return