import torch

import triton
import triton.language as tl
import math
import torch.nn.functional as F

TESLA = "Tesla" in torch.cuda.get_device_name(0)


if triton.__version__ >= "2.1.0":

    @triton.jit
    def _fwd_kernel(
        Q,
        K,
        V,
        sm_scale,
        seq_len,
        Out,
        stride_b,
        stride_s,
        stride_h,
        stride_d,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cur_batch = tl.program_id(0)
        cur_head = tl.program_id(1)
        start_m = tl.program_id(2)

        # initialize offsets
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        off_q = (
            cur_batch * stride_b
            + offs_m[:, None] * stride_s
            + cur_head * stride_h
            + offs_d[None, :] * stride_d
        )

        q = tl.load(Q + off_q, mask=offs_m[:, None] < seq_len, other=0.0)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            off_k = cur_batch * stride_b + (start_n + offs_n[None, :]) * stride_s + cur_head * stride_h + offs_d[:, None] * stride_d
            k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < seq_len, other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            qk = tl.where((start_n + offs_n[None, :]) < seq_len , qk, float("-inf"))

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            off_v = cur_batch * stride_b + (start_n + offs_n[:, None]) * stride_s + cur_head * stride_h + offs_d[None, :] * stride_d
            v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0)
            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
        # initialize pointers to output
        out_ptrs = Out + off_q
        tl.store(out_ptrs, acc, mask=offs_m[:, None] < seq_len)
        return

    @torch.no_grad()
    def flash_attention_fwd(
        q, k, v, o,
    ):
        BLOCK = 128 if not TESLA else 64
        # shape constraints
        batch_size, seq_len, head_num, head_dim = q.shape
        assert head_dim in {16, 32, 64, 128, 256}
        sm_scale = 1.0 / (head_dim ** 0.5)  # 计算scale系数
        grid = (batch_size, head_num, triton.cdiv(seq_len, BLOCK))  # batch, head,
        num_warps = 4 if head_dim <= 64 else 8
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            seq_len,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=1,
        )
        return

else:
    raise Exception("error triton version!")


def torch_att(q, k, v):
    batch_size = q.shape[0]
    seq_len = q.shape[1]
    head_dim = q.shape[-1]
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    scale = head_dim ** -0.5
    attn = ((q * scale) @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    out = attn @ v
    out =  out.transpose(1,2).contiguous()
    return out


def test():
    import torch
    import numpy as np

    B, L, H, D = 4, 1025, 7, 128
    dtype = torch.float16
    Z = 1
    q = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    k = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    v = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)
    o = torch.empty((B, L, H, D), dtype=dtype, device="cuda").normal_(mean=0.1, std=0.2)

    torch_out = torch_att(q, k, v)
    import time
    torch.cuda.synchronize()
    a = time.time()
    for i in range(100):
        context_attention_fwd(
            q, k, v, o
        )
        # o = torch_att(q, k, v)
    torch.cuda.synchronize()
    b = time.time()
    # print(o.shape, torch_out.shape)
    print((b - a) / 100 * 1000)

    print("max ", torch.max(torch.abs(torch_out - o)))
    print("mean ", torch.mean(torch.abs(torch_out - o)))
    assert torch.allclose(torch_out, o, atol=1e-2, rtol=0)