import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_kv_fp8(
    K,
    Dest_loc,
    Out,
    Out_scale,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    stride_o_scale_bs,
    stride_o_scale_h,
    stride_o_scale_d,
    head_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index).to(tl.int64)

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    # to fp8
    k = tl.load(k_ptrs, mask=offs_h[:, None] < head_num, other=0.0)
    max_k = tl.maximum(tl.max(tl.abs(k), axis=1), 1e-12)
    k_scale = (max_k / FP8_MAX).to(k.dtype)
    k_fp8 = tl.clamp(k / k_scale, min=FP8_MIN, max=FP8_MAX).to(tl.float8e4nv)

    # save kv_scale
    offs_k_scale = tl.arange(0, 1)
    o_scale_ptrs = Out_scale + dest_index * stride_o_scale_bs + stride_o_scale_d * offs_k_scale
    tl.store(o_scale_ptrs, k_scale)

    tl.store(o_ptrs, k_fp8, mask=offs_h[:, None] < head_num)
    return


@torch.no_grad()
def destindex_copy_kv_fp8(K, DestLoc, Out, Out_scale):
    seq_len = DestLoc.shape[0]
    head_num = K.shape[1]
    head_dim = K.shape[2]
    assert K.shape[1] == Out.shape[1] and K.shape[2] == Out.shape[2]
    BLOCK_HEAD = triton.next_power_of_2(head_num)
    grid = (seq_len,)
    num_warps = 1

    _fwd_kernel_destindex_copy_kv_fp8[grid](
        K,
        DestLoc,
        Out,
        Out_scale,
        K.stride(0),
        K.stride(1),
        K.stride(2),
        Out.stride(0),
        Out.stride(1),
        Out.stride(2),
        Out_scale.stride(0),
        Out_scale.stride(1),
        Out_scale.stride(2),
        head_num,
        BLOCK_DMODEL=head_dim,
        BLOCK_HEAD=BLOCK_HEAD,
        FP8_MIN=torch.finfo(torch.float8_e4m3fn).min,
        FP8_MAX=torch.finfo(torch.float8_e4m3fn).max,
        num_warps=num_warps,
        num_stages=1,
    )
    return


if __name__ == "__main__":
    import torch.nn.functional as F

    B, N_CTX, H, HEAD_DIM = 32, 1024, 16, 128
    dtype = torch.bfloat16
    NUM = 20
    dest_loc = torch.arange(NUM).cuda()
    kv = torch.randn((len(dest_loc), H, HEAD_DIM), dtype=dtype).cuda()
    out = torch.zeros((B * N_CTX, H, HEAD_DIM + 2), dtype=torch.uint8).cuda()

    fp8_type = torch.float8_e4m3fn
    O_ = out[:, :, :HEAD_DIM].view(fp8_type)
    O_scale = out[:, :, -2:].view(dtype)
    destindex_copy_kv_fp8(kv, dest_loc, O_, O_scale)

    cos = F.cosine_similarity(O_[:NUM].to(dtype) * O_scale[:NUM], kv).mean()
    print(cos)
