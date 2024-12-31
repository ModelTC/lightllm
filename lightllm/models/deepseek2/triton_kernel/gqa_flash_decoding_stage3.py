import os
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage3(
    B_Seqlen,
    Mid_O,  # [batch, head, seq_block_num, head_dim]
    Mid_O_LogExpSum,  # [batch, head, seq_block_num]
    Out,  # [batch, head, head_dim]
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    block_seq,
    BLOCK_DMODEL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + block_seq - 1) // block_seq

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in tl.range(0, block_n_size, 1, num_stages=NUM_STAGES):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)

        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return


@torch.no_grad()
def flash_decode_stage3(mid_out, mid_out_logexpsum, B_Seqlen, Out, **run_config):
    if run_config:
        BLOCK_SEQ = run_config["STAGE2_BLOCK_SEQ"]
        num_warps = run_config["stage3_num_warps"]
        num_stages = run_config["stage3_num_stages"]

    Lk = mid_out.shape[-1]
    assert Lk in {16, 32, 64, 128, 256, 512}
    batch, head_num = mid_out.shape[0], mid_out.shape[1]
    grid = (head_num, batch)

    _fwd_kernel_flash_decode_stage3[grid](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        Out,
        *mid_out.stride(),
        *mid_out_logexpsum.stride(),
        *Out.stride(),
        block_seq=BLOCK_SEQ,
        BLOCK_DMODEL=Lk,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return
