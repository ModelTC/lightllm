"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger
from lightllm.common.vllm_kernel import _custom_ops as ops
from lightllm.utils.vllm_utils import direct_register_custom_op

logger = init_logger(__name__)


@triton.jit
def moe_align_kernel(
    topk_ids_ptr,
    topk_ids_stride_m,
    topk_ids_stride_n,
    topk_m,
    topk_n,
    out_ptr,
    out_stride_m,
    out_stride_n,
    out_m,
    out_n,
    TOPK_BLOCK_M: tl.constexpr,
    TOPK_BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    m_range = pid * TOPK_BLOCK_M + tl.arange(0, TOPK_BLOCK_M)
    n_range = tl.arange(0, TOPK_BLOCK_N)

    topk_ptr = topk_ids_ptr + m_range[:, None] * topk_ids_stride_m + n_range[None, :]
    topk_datas = tl.load(topk_ptr, mask=(m_range[:, None] < topk_m) & (n_range[None, :] < topk_n), other=-1)
    write_datas = tl.where(topk_datas != -1, 1, 0)

    tl.store(
        out_ptr + topk_datas * out_stride_m + m_range[:, None],
        write_datas,
        mask=(m_range[:, None] < topk_m) & (n_range[None, :] < topk_n),
    )


def moe_align(topk_ids: torch.Tensor, out: torch.Tensor):
    """
    topk_ids is tensor like [[0, 1, 2], [0, 3, 1], [3, 1, 4]] shape is [token_num, topk_num]
    out is tensor is shape with [expert_num, token_num]
    out need fill 0 first, and then, fill the value to 1 in selected token loc.
    when expert_num is 5 and token_num is 3.
    topk_ids = [[0, 1, 2], [0, 3, 1], [3, 1, 4]]
    out tensor will be:
    [
    [1, 1, 0,],
    [1, 1, 1,],
    [1, 0, 0,],
    [0, 1, 1,],
    [0, 0, 1,]
    ]
    """
    TOPK_BLOCK_M = 64

    token_num, topk = topk_ids.shape
    expert_num = out.shape[0]
    assert out.shape[1] == token_num
    grid = (triton.cdiv(token_num, TOPK_BLOCK_M),)
    moe_align_kernel[grid](
        topk_ids,
        topk_ids.stride(0),
        topk_ids.stride(1),
        token_num,
        topk,
        out,
        out.stride(0),
        out.stride(1),
        expert_num,
        token_num,
        TOPK_BLOCK_M=TOPK_BLOCK_M,
        TOPK_BLOCK_N=triton.next_power_of_2(topk),
        num_warps=8,
        num_stages=1,
    )


# one grid handle a expert data
@triton.jit
def moe_align1_kernel(
    experts_info_ptr,
    experts_stride_m,
    experts_stride_n,
    experts_info_m,
    experts_info_n,
    expert_token_num_ptr,
    TOKEN_BLOCK_N: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    n_range = tl.arange(0, TOKEN_BLOCK_N)

    expert_data = tl.load(experts_info_ptr + pid * experts_stride_m + n_range, mask=n_range < experts_info_n, other=0)
    cumsum_expert_data = tl.cumsum(expert_data)

    tl.store(expert_token_num_ptr + pid, tl.max(cumsum_expert_data))
    tl.store(
        experts_info_ptr + pid * experts_stride_m + cumsum_expert_data - 1,
        n_range,
        mask=(expert_data == 1) & (n_range < experts_info_n),
    )


def moe_align1(experts_info: torch.Tensor, exports_token_num: torch.Tensor):
    """
    experts_info is tensor shape [expert_num, token_num],
    exports_token_num is out tensor, will get expert need handle token num.

    experts_info will change inplace.
    demo:
    expert_num = 2, token_num = 4
    experts_info = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    will get:
    experts_info = [
        [0, 2, x, x],  # mark x， data will not be used future
        [1, 3, x, x]
    ]

    exports_token_num = [2, 2]
    """
    expert_num, token_num = experts_info.shape
    assert token_num < 8072, "need split to handle seq len too long"
    assert exports_token_num.shape[0] == expert_num
    TOKEN_BLOCK_N = triton.next_power_of_2(token_num)
    grid = (expert_num,)
    moe_align1_kernel[grid](
        experts_info,
        experts_info.stride(0),
        experts_info.stride(1),
        expert_num,
        token_num,
        exports_token_num,
        TOKEN_BLOCK_N=TOKEN_BLOCK_N,
        num_warps=8,
        num_stages=1,
    )
