import triton
import triton.language as tl


def init_req_to_token_indexes(
    req_to_token_indexs, b_req_idx, b_seq_len, b_ready_cache_len, max_len_in_batch, alloc_mem_index
):
    start_index = 0
    b_seq_len_numpy = b_seq_len.cpu().numpy()
    b_ready_cache_len_numpy = b_ready_cache_len.cpu().numpy()
    b_req_idx_numpy = b_req_idx.cpu().numpy()
    for i in range(len(b_seq_len)):
        cur_seq_len = b_seq_len_numpy[i]
        cur_ready_cache_len = b_ready_cache_len_numpy[i]
        req_to_token_indexs[b_req_idx_numpy[i], cur_ready_cache_len:cur_seq_len] = alloc_mem_index[
            start_index : start_index + cur_seq_len - cur_ready_cache_len
        ]
        start_index += cur_seq_len - cur_ready_cache_len
    return


@triton.jit
def _fwd_gather_kvs(
    tgt_ptr,
    tgt_stride_token,
    tgt_stride_hid,
    src_ptr,
    src_stride_token,
    src_stride_hid,
    idx_ptr,
    idx_stride,
    HIDDEN_DIM: tl.constexpr,
):
    cur_len = tl.program_id(0)
    cur_idx = tl.load(idx_ptr + cur_len * idx_stride)
    hid_offset = tl.arange(0, HIDDEN_DIM)
    tgt_offset = tgt_ptr + cur_len * tgt_stride_token + tgt_stride_hid * hid_offset
    src_offset = src_ptr + cur_idx * src_stride_token + src_stride_hid * hid_offset
    tl.store(tgt_offset, tl.load(src_offset))


def gather_kvs_from_idx(tgt, src, idx):
    grid = (len(idx),)
    HIDDEN_DIM = tgt.shape[1]
    num_warps = 1
    assert len(tgt.shape) == 2
    assert len(src.shape) == 2
    assert len(tgt) == len(idx)
    assert idx.max() <= src.shape[0]

    # torch.save((tgt, src, idx), 'test.pt')
    _fwd_gather_kvs[grid](
        tgt,
        tgt.stride(0),
        tgt.stride(1),
        src,
        src.stride(0),
        src.stride(1),
        idx,
        idx.stride(0),
        HIDDEN_DIM=HIDDEN_DIM,
        num_warps=num_warps,
        num_stages=1,
    )


@triton.jit
def _fwd_scatter_kvs(
    tgt_ptr,
    tgt_stride_token,
    tgt_stride_hid,
    src_ptr,
    src_stride_token,
    src_stride_hid,
    idx_ptr,
    idx_stride,
    HIDDEN_DIM: tl.constexpr,
):
    cur_len = tl.program_id(0)
    cur_idx = tl.load(idx_ptr + cur_len * idx_stride)
    hid_offset = tl.arange(0, HIDDEN_DIM)
    tgt_offset = tgt_ptr + cur_idx * tgt_stride_token + tgt_stride_hid * hid_offset
    src_offset = src_ptr + cur_len * src_stride_token + src_stride_hid * hid_offset
    tl.store(tgt_offset, tl.load(src_offset))


def scatter_kvs_to_idx(tgt, src, idx):
    grid = (len(idx),)
    HIDDEN_DIM = tgt.shape[1]
    num_warps = 1
    _fwd_scatter_kvs[grid](
        tgt,
        tgt.stride(0),
        tgt.stride(1),
        src,
        src.stride(0),
        src.stride(1),
        idx,
        idx.stride(0),
        HIDDEN_DIM=HIDDEN_DIM,
        num_warps=num_warps,
        num_stages=1,
    )
