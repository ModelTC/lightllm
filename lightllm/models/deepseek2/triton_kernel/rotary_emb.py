import torch

import triton
import triton.language as tl


@triton.jit
def _rotary_kernel_q(
    Q,
    Cos,
    Sin,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    HEAD_Q,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_head_index = tl.program_id(0)
    if cur_head_index >= HEAD_Q:
        return
    cur_seq_index = tl.program_id(1)

    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2) * 2
    dim_range1 = dim_range0 + 1

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs + cur_head_index * stride_qh + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs + cur_head_index * stride_qh + dim_range1[None, None, :] * stride_qd
    )
    mask = cur_seq_range[:, None, None] < max_total_len

    cos_range = tl.arange(0, BLOCK_DMODEL // 2)
    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + cos_range[None, None, :] * stride_cosd

    q0 = tl.load(
        Q + off_q0,
        mask=mask,
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=mask,
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out0 = q0 * cos - q1 * sin
    out1 = q0 * sin + q1 * cos

    tl.store(Q + off_q0, out0, mask=mask)
    tl.store(Q + off_q1, out1, mask=mask)
    return


@triton.jit
def _rotary_kernel_k(
    K,
    Cos,
    Sin,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    HEAD_K,  # HEAD_K is 1.
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_seq_index = tl.program_id(0)

    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    dim_range0 = tl.arange(0, BLOCK_DMODEL // 2) * 2
    dim_range1 = dim_range0 + 1

    cos_range = tl.arange(0, BLOCK_DMODEL // 2)
    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + cos_range[None, None, :] * stride_cosd

    off_k0 = cur_seq_range[:, None, None] * stride_kbs + dim_range0[None, None, :] * stride_kd
    off_k1 = cur_seq_range[:, None, None] * stride_kbs + dim_range1[None, None, :] * stride_kd

    off_dimcos_sin = cur_seq_range[:, None, None] * stride_cosbs + cos_range[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len),
        other=0.0,
    )

    cos = tl.load(Cos + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)
    sin = tl.load(Sin + off_dimcos_sin, mask=cur_seq_range[:, None, None] < max_total_len, other=0.0)

    out_k0 = k0 * cos - k1 * sin
    out_k1 = k0 * sin + k1 * cos

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len),
    )
    return


@torch.no_grad()
def rotary_emb_fwd(q, k, cos, sin):
    total_len = q.shape[0]
    head_num_q, head_num_k = q.shape[1], k.shape[1]
    head_dim = q.shape[2]
    assert q.shape[0] == cos.shape[0] and q.shape[0] == sin.shape[0], f"q shape {q.shape} cos shape {cos.shape}"
    assert k.shape[0] == cos.shape[0] and k.shape[0] == sin.shape[0], f"k shape {k.shape} cos shape {cos.shape}"

    BLOCK_SEQ = 16
    BLOCK_HEAD = 2
    if head_dim >= 128:
        num_warps = 8
    else:
        num_warps = 4
    grid = (triton.next_power_of_2(head_num_q), triton.cdiv(total_len, BLOCK_SEQ))
    _rotary_kernel_q[grid](
        q,
        cos,
        sin,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        total_len,
        head_num_q,
        BLOCK_HEAD=BLOCK_HEAD,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    grid = (triton.cdiv(total_len, BLOCK_SEQ),)
    _rotary_kernel_k[grid](
        k,
        cos,
        sin,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        cos.stride(0),
        cos.stride(1),
        sin.stride(0),
        sin.stride(1),
        total_len,
        head_num_k,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DMODEL=head_dim,
        num_warps=num_warps,
        num_stages=1,
    )
    return
