import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage2_padding(
    mid_out_logics,
    stride_mid_out_logics_b,
    stride_mid_out_logics_h,
    stride_mid_out_logics_s,
    KV,
    stride_kv_bs,
    stride_kv_h,
    stride_kv_d,
    Req_to_tokens,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    B_req_idx,
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    gqa_group_size,
    head_dim,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    SPLIT_K_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    seq_start_block = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    cur_batch = tl.program_id(2)

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, Q_HEAD_NUM)
    head_mask = cur_q_head_range < (cur_kv_head + 1) * gqa_group_size

    offs_split_d = tl.arange(0, SPLIT_K_DIM)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)

    block_ok = tl.where(cur_batch_end_index - cur_batch_start_index <= 0, 0, 1)

    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_SEQ)
    for start_n in range(0, block_ok, 1):
        offs_n_new = start_n * BLOCK_SEQ + offs_n
        n_seq_mask = offs_n_new < cur_batch_end_index
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + offs_n_new,
            mask=n_seq_mask,
            other=0,
        )
        att_ptrs = (
            mid_out_logics + cur_batch * stride_mid_out_logics_b + cur_q_head_range[:, None] * stride_mid_out_logics_h
        )
        att_value = tl.load(
            att_ptrs + offs_n_new[None, :], mask=head_mask[:, None] & n_seq_mask[None, :], other=float("-inf")
        )
        max_logics = tl.max(att_value, axis=1)
        tmp_exp = tl.exp(att_value - max_logics[:, None])
        sum_exp = tl.sum(tmp_exp, axis=1)
        prob = tmp_exp / sum_exp[:, None]
        prob = prob.to(KV.dtype.element_ty)
        for index in tl.range(0, head_dim // SPLIT_K_DIM, num_stages=NUM_STAGES):
            off_kv = kv_loc[:, None] * stride_kv_bs + offs_split_d[None, :] + index * SPLIT_K_DIM
            cur_kv = tl.load(KV + stride_kv_h * cur_kv_head + off_kv, mask=n_seq_mask[:, None], other=0.0)
            tmp_o = tl.dot(prob, cur_kv)
            off_mid_o = (
                cur_batch * stride_mid_ob
                + cur_q_head_range[:, None] * stride_mid_oh
                + seq_start_block * stride_mid_os
                + offs_split_d[None, :]
                + index * SPLIT_K_DIM
            )
            # print(Mid_O + off_mid_o)
            tl.store(Mid_O + off_mid_o, tmp_o, mask=head_mask[:, None])

        tl.store(
            Mid_O_LogExpSum + cur_batch * stride_mid_o_eb + stride_mid_o_eh * cur_q_head_range + seq_start_block,
            max_logics + tl.log(sum_exp),
            mask=head_mask,
        )
    return


@torch.no_grad()
def flash_decode_stage2(
    mid_out_logics: torch.Tensor,
    kv: torch.Tensor,
    req_to_token_indexs: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_in_batch,
    mid_o: torch.Tensor,
    mid_o_logexpsum: torch.Tensor,
    rope_head_dim: int,
    **run_config
):
    if run_config:
        BLOCK_SEQ = run_config["STAGE2_BLOCK_SEQ"]
        SPLIT_K_DIM = run_config["STAGE2_SPLIT_K_DIM"]
        num_warps = run_config["stage2_num_warps"]
        num_stages = run_config["stage2_num_stages"]

    head_dim = kv.shape[-1] - rope_head_dim
    assert head_dim % SPLIT_K_DIM == 0
    batch, q_head_num, kv_head_num = mid_out_logics.shape[0], mid_out_logics.shape[1], kv.shape[1]

    grid = (triton.cdiv(max_len_in_batch, BLOCK_SEQ), kv_head_num, batch)
    gqa_group_size = q_head_num // kv_head_num
    _fwd_kernel_flash_decode_stage2_padding[grid](
        mid_out_logics,
        mid_out_logics.stride(0),
        mid_out_logics.stride(1),
        mid_out_logics.stride(2),
        kv,
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        req_to_token_indexs,
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        b_req_idx,
        b_seq_len,
        mid_o,  # [batch, head, seq_block_num, head_dim]
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        mid_o_logexpsum,  # [batch, head, seq_block_num]
        mid_o_logexpsum.stride(0),
        mid_o_logexpsum.stride(1),
        mid_o_logexpsum.stride(2),
        gqa_group_size,
        head_dim,
        Q_HEAD_NUM=max(16, triton.next_power_of_2(q_head_num)),
        BLOCK_SEQ=BLOCK_SEQ,
        SPLIT_K_DIM=SPLIT_K_DIM,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )
    return
