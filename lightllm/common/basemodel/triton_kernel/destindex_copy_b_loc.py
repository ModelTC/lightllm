import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_b_loc(
    B_Loc, B_Loc_idx, Dest_loc, memindex,
    stride_b_loc_0, stride_b_loc_1
):
    cur_index = tl.program_id(0)
    cur_b_loc_idx = tl.load(B_Loc_idx + cur_index)
    cur_mem_index = tl.load(memindex + cur_index)
    cur_dest_loc = tl.load(Dest_loc + cur_index)
    b_loc_offset = B_Loc + cur_b_loc_idx * stride_b_loc_0 + cur_dest_loc * stride_b_loc_1
    tl.store(b_loc_offset, cur_mem_index)
    return


@torch.no_grad()
def destindex_copy_b_loc(B_Loc, B_Loc_idx, DestLoc, memindex):
    seq_len = DestLoc.shape[0]
    assert DestLoc.shape[0] == memindex.shape[0] and B_Loc_idx.shape[0] == DestLoc.shape[0]
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_b_loc[grid](
        B_Loc, B_Loc_idx, DestLoc, memindex,
        B_Loc.stride(0), B_Loc.stride(1),
        num_warps=num_warps,
        num_stages=1,
    )
    return

def test1():
    import time
    import numpy as np
    B, N_CTX = 16, 1024
    b_loc_idx = np.random.choice(np.arange(0, B * 10), B, replace=False)
    b_loc_idx = torch.tensor(b_loc_idx, dtype=torch.int32, device='cuda')
    b_loc = torch.zeros((B * 10 , N_CTX), dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros((B,), dtype=torch.int32, device="cuda")

    for i in range(B):
        b_seq_len[i] = i + 1
    memindex = torch.arange(2, B + 2, dtype=torch.int32, device="cuda")


    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        for i in range(B):
            b_loc[b_loc_idx[i]][b_seq_len[i]] = memindex[i]
    torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)


    for _ in range(10):
        destindex_copy_b_loc(b_loc, b_loc_idx, b_seq_len, memindex)
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        destindex_copy_b_loc(b_loc, b_loc_idx, b_seq_len, memindex)
    torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)
    # for i in range(B):
    #     print(b_loc[b_loc_idx[i]][b_seq_len[i]])

if __name__ == '__main__':
    test1()
