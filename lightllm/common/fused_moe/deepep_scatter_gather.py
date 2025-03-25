import random
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict


@triton.jit
def _fwd_kernel_ep_scatter_1(
    tmp,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    num_recv_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    m_indices,
    topk_row: tl.constexpr,
    topk_col: tl.constexpr,
    num_experts: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, num_experts)
    tokens_per_expert = tl.load(num_recv_tokens_per_expert + offset_cumsum)
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_token_num, BLOCK_E, num_stages=4):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    tmp,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    num_recv_tokens_per_expert,
    expert_start_loc,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    m_indices,
    topk_row: tl.constexpr,
    topk_col: tl.constexpr,
    num_experts: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)

    for start_topk in tl.range(0, topk_col, 1, num_stages=4):
        topk = tl.load(recv_topk + token_id * recv_topk_stride0 + start_topk)
        if topk >= 0:
            cur_index = tl.atomic_add(tmp + topk, 1)
            tl.store(output_index + token_id * output_index_stride0 + start_topk, cur_index)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    offset_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = offset_in_s < SCALE_HIDDEN_SIZE

    to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
    to_copy_s = tl.load(recv_x_scale + token_id * recv_x_scale_stride0 + offset_in_s, mask=mask_s)

    for start_topk in range(0, topk_col):
        cur_expert = tl.load(recv_topk + token_id * recv_topk_stride0 + start_topk)
        if cur_expert >= 0:
            start_ = tl.load(expert_start_loc + cur_expert)
            dst = tl.load(output_index + token_id * output_index_stride0 + start_topk) + start_

            output_tensor_ptr = output_tensor + dst * output_tensor_stride0
            output_tensor_scale_ptr = output_tensor_scale + dst * output_tensor_scale_stride0
            tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
            tl.store(output_tensor_scale_ptr + offset_in_s, to_copy_s, mask=mask_s)


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
):
    BLOCK_E = 128  # token num of per expert is aligned to 128
    BLOCK_D = 128  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]  # 获取num_recv_tokens_per_expert的元素个数
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts

    tmp = torch.zeros((num_experts,), device="cuda", dtype=torch.int32)

    _fwd_kernel_ep_scatter_1[(grid,)](
        tmp,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        m_indices,
        topk_row=recv_topk.shape[0],
        topk_col=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        BLOCK_E=BLOCK_E,
        BLOCK_D=BLOCK_D,
    )

    grid = recv_topk.shape[0]

    _fwd_kernel_ep_scatter_2[(grid,)](
        tmp,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0),
        recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0),
        output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        m_indices,
        topk_row=recv_topk.shape[0],
        topk_col=recv_topk.shape[1],
        num_experts=num_experts,
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=hidden_size // BLOCK_D,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size // BLOCK_D),
        BLOCK_E=BLOCK_E,
        BLOCK_D=BLOCK_D,
    )

    return


@triton.jit
def _fwd_kernel_ep_gather(
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    expert_start_loc,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_col: tl.constexpr,
    stride_tokens: tl.constexpr,
    num_tokens: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    token_start = tl.program_id(1)
    off_d = tl.arange(0, BLOCK_D)
    for cur_token in range(token_start, num_tokens, stride_tokens):
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
        for start_topk in range(0, topk_col):
            expert_id = tl.load(recv_topk + cur_token * recv_topk_stride0 + start_topk)
            if expert_id >= 0:
                expert_offset = tl.load(input_index + cur_token * input_index_stride0 + start_topk)
                cur_expert_start = tl.load(expert_start_loc + expert_id)
                acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + start_topk)
                tmp = tl.load(
                    input_tensor
                    + (cur_expert_start + expert_offset) * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                accumulator += tmp.to(tl.float32) * acc_weight
        tl.store(
            output_tensor + cur_token * output_tensor_stride0 + cur_block * BLOCK_D + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
):
    BLOCK_D = 128  # block size of quantization
    BLOCK_T = 128  # token blocks
    num_warps = 4
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    grid_t = triton.cdiv(num_tokens, BLOCK_T)
    grid = (triton.cdiv(hidden_size, BLOCK_D), grid_t)
    _fwd_kernel_ep_gather[grid](
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        expert_start_loc,
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_col=recv_topk.shape[1],
        stride_tokens=grid_t,
        num_tokens=num_tokens,
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return
