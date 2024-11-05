import torch
import triton
import triton.language as tl


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


# INFO 11-05 10:25:12 [impl.py:86] gather_kvs shape torch.Size([4, 2048]) dtype torch.bfloat16 device cuda:1
# INFO 11-05 10:25:12 [impl.py:87] kv_buffer shape torch.Size([400000, 16, 128]) dtype torch.bfloat16 device cuda:1
# INFO 11-05 10:25:12 [impl.py:88] alloc_idx shape torch.Size([4]) dtype torch.int32 device cuda:1


if __name__ == "__main__":
    torch.cuda.set_device(1)
    tgt = torch.zeros((4, 2048), dtype=torch.bfloat16, device="cuda")
    src = torch.zeros((400000, 16, 128), dtype=torch.bfloat16, device="cuda")
    idx = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    print(tgt.cuda())
    scatter_kvs_to_idx(src, tgt, idx)
