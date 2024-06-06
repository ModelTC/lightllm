import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_init_att_window_info(
    b_seq_len, b_start_loc_window, b_att_seq_len, 
    sliding_window
):
    cur_index = tl.program_id(0)
    cur_seq_len = tl.load(b_seq_len + cur_index)

    b_start_loc_window_data = tl.maximum(cur_seq_len - sliding_window, 0)
    b_att_seq_len_data = tl.minimum(cur_seq_len, sliding_window)

    tl.store(b_start_loc_window + cur_index, b_start_loc_window_data)
    tl.store(b_att_seq_len + cur_index, b_att_seq_len_data)
    return


@torch.no_grad()
def init_att_window_info_fwd(
    batch_size, b_seq_len, b_start_loc_window, b_att_seq_len, sliding_window):
    # shape constraints
    assert batch_size == b_seq_len.shape[0] == b_start_loc_window.shape[0] == b_att_seq_len.shape[0]

    grid = (batch_size,)
    num_warps = 1

    _fwd_kernel_init_att_window_info[grid](
        b_seq_len, b_start_loc_window, b_att_seq_len, 
        sliding_window=sliding_window,
        num_warps=num_warps,
        num_stages=1,
    )
    return