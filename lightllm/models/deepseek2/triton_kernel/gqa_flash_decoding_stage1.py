import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage1_padding(
    Q,
    KV,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    logics,  # [batch, head, seq_len]
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    stride_logics_b,
    stride_logics_h,
    stride_logics_s,
    gqa_group_size,
    head_dim,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    SPLIT_K_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    seq_start_block = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_batch = tl.program_id(2)

    cur_q_head_offs = tl.arange(0, Q_HEAD_NUM)
    cur_q_head_range = cur_kv_head * gqa_group_size + cur_q_head_offs
    head_mask = cur_q_head_range < (cur_kv_head + 1) * gqa_group_size

    offs_split_d = tl.arange(0, SPLIT_K_DIM)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    off_q = cur_batch * stride_q_b + cur_q_head_range[:, None] * stride_q_h + offs_split_d[None, :]
    off_o = cur_batch * stride_logics_b + cur_q_head_range[:, None] * stride_logics_h
    block_n_size = (
        tl.where(
            cur_batch_end_index - cur_batch_start_index <= 0,
            0,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
        )
        // BLOCK_N
    )

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    for start_n in tl.range(0, block_n_size, 1, num_stages=2):
        offs_n_new = start_n * BLOCK_N + offs_n
        seq_n_mask = offs_n_new < cur_batch_end_index
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=seq_n_mask,
            other=0,
        )
        att_value = tl.zeros([Q_HEAD_NUM, BLOCK_N], dtype=tl.float32)

        att_value = q_dot_k(
            Q,
            KV,
            kv_loc,
            att_value,
            stride_kv_bs,
            stride_kv_h,
            head_dim,
            cur_kv_head,
            head_mask,
            offs_split_d,
            off_q,
            seq_n_mask,
            SPLIT_K_DIM,
            NUM_STAGE,
        )

        att_value *= sm_scale
        tl.store(logics + off_o + offs_n_new[None, :], att_value, mask=head_mask[:, None] & seq_n_mask[None, :])
    return


@triton.jit
def q_dot_k(
    Q,
    KV,
    kv_loc,
    att_value,
    stride_kv_bs,
    stride_kv_h,
    head_dim,
    cur_kv_head,
    head_mask,
    offs_split_d,
    off_q,
    seq_n_mask,
    SPLIT_K_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    off_kv = kv_loc[None, :] * stride_kv_bs + cur_kv_head * stride_kv_h + offs_split_d[:, None]
    for index in tl.range(head_dim // SPLIT_K_DIM, num_stages=NUM_STAGE):
        q = tl.load(Q + off_q + index * SPLIT_K_DIM, mask=head_mask[:, None], other=0.0)
        kv = tl.load(KV + off_kv + index * SPLIT_K_DIM, mask=seq_n_mask[None, :], other=0.0)
        att_value += tl.dot(q, kv)
    return att_value


@torch.no_grad()
def flash_decode_stage1(
    q: torch.Tensor,
    kv: torch.Tensor,
    Req_to_tokens: torch.Tensor,
    B_req_idx: torch.Tensor,
    B_Seqlen: torch.Tensor,
    max_len_in_batch: torch.Tensor,
    mid_out_logics: torch.Tensor,
    softmax_scale: float,
    **run_config,
):
    if run_config:
        BLOCK_SEQ = run_config["STAGE1_BLOCK_SEQ"]
        BLOCK_N = run_config["STAGE1_BLOCK_N"]
        SPLIT_K_DIM = run_config["STAGE1_SPLIT_K_DIM"]
        num_warps = run_config["stage1_num_warps"]
        num_stages = run_config["stage1_num_stages"]

    head_dim = q.shape[-1]  # nope_dim + rope_dim
    assert BLOCK_SEQ % BLOCK_N == 0
    assert head_dim % SPLIT_K_DIM == 0
    assert head_dim == 512 + 64

    batch, q_head_num = B_req_idx.shape[0], q.shape[1]
    kv_head_num = kv.shape[1]
    grid = (triton.cdiv(max_len_in_batch, BLOCK_SEQ), kv_head_num, batch)
    gqa_group_size = q_head_num // kv_head_num
    _fwd_kernel_flash_decode_stage1_padding[grid](
        q,
        kv,
        softmax_scale,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        mid_out_logics,  # [batch, head, seq_len]
        *Req_to_tokens.stride(),
        *q.stride(),
        *kv.stride(),
        *mid_out_logics.stride(),
        gqa_group_size,
        head_dim,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(q_head_num)),
        BLOCK_SEQ=BLOCK_SEQ,
        SPLIT_K_DIM=SPLIT_K_DIM,
        BLOCK_N=BLOCK_N,
        NUM_STAGE=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
