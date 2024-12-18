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
    topk_m,
    topk_n,
    out_ptr,
    out_stride_m,
    out_stride_n,
    TOPK_BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    m_range = pid * TOPK_BLOCK_M + tl.arange(0, TOPK_BLOCK_M)

    topk_ptr = topk_ids_ptr + m_range
    topk_datas = tl.load(topk_ptr, mask=m_range < topk_m * topk_n, other=-1)
    write_datas = tl.where(topk_datas != -1, 1, 0)

    tl.store(
        out_ptr + topk_datas * out_stride_m + m_range,
        write_datas,
        mask=m_range < topk_m * topk_n,
    )


def moe_align(topk_ids: torch.Tensor, out: torch.Tensor):
    """
    topk_ids is tensor like [[0, 1, 2], [0, 3, 1], [3, 1, 4]] shape is [token_num, topk_num],
    the topk_ids needs to be in contiguous memory.
    out is tensor is shape with [expert_num, token_num * topk_num]
    out need fill 0 first, and then, fill the value to 1 in selected token loc.
    when expert_num is 5 and token_num is 3. topk_num is 3.
    topk_ids = [[0, 1, 2], [0, 3, 1], [3, 1, 4]]
    out tensor will be:
    [
    [1, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]
    """
    TOPK_BLOCK_M = 128

    token_num, topk = topk_ids.shape
    assert out.shape[1] == token_num * topk
    grid = (triton.cdiv(token_num, TOPK_BLOCK_M),)
    moe_align_kernel[grid](
        topk_ids,
        token_num,
        topk,
        out,
        out.stride(0),
        out.stride(1),
        TOPK_BLOCK_M=TOPK_BLOCK_M,
        num_warps=4,
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


def moe_align1(experts_info: torch.Tensor, exports_token_num: torch.Tensor, topk: int):
    """
    experts_info is tensor shape [expert_num, token_num * topk],
    exports_token_num is out tensor, will get expert need handle token num.

    experts_info will change inplace.
    demo:
    topids = [[0, 1], [1, 3]]
    expert_num = 4, token_num = 2, topk = 2
    experts_info = [
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]
    will get:
    experts_info = [
        [0, x, x, x],  # mark x， data will not be used future
        [1, 2, x, x],
        [3, x, x, x],
        [x, x, x, x]
    ]

    exports_token_num = [1, 2, 1, 0]
    """
    expert_num, token_num_mul_topk = experts_info.shape
    assert token_num_mul_topk < 8072 * 2, "need split to handle seq len too long"
    assert exports_token_num.shape[0] == expert_num
    TOKEN_BLOCK_N = triton.next_power_of_2(token_num_mul_topk)
    grid = (expert_num,)
    moe_align1_kernel[grid](
        experts_info,
        experts_info.stride(0),
        experts_info.stride(1),
        expert_num,
        token_num_mul_topk,
        exports_token_num,
        TOKEN_BLOCK_N=TOKEN_BLOCK_N,
        num_warps=8,
        num_stages=1,
    )


# DEVICE = triton.runtime.driver.active.get_active_torch_device()


# @triton.jit
# def grouped_matmul_kernel(
#     token_ptr, # [token_num, hidden_dim]
#     weights_ptrs, # [expert_num]
#     weights_kn, # [expert_num, 2]
#     expert_to_token_num, # [expert_num]
#     expert_to_token_index, # [expert_num, token_num]
#     expert_num, # int
#     k, # int
#     n, # int
#     out, # [token_num, topk, n]
#     out_stride_0,
#     out_stride_1,
#     out_stride_2,
#     # number of virtual SM
#     NUM_SM: tl.constexpr,
#     # tile sizes
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     tile_idx = tl.program_id(0)
#     last_problem_end = 0
#     tl_calcu_type = token_ptr.dtype.element_ty
#     for expert_id in range(expert_num):
#         # get the gemm size of the current problem
#         cur_m = tl.load(expert_to_token_num + expert_id)
#         num_m_tiles = tl.cdiv(cur_m, BLOCK_SIZE_M)
#         num_n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
#         num_tiles = num_m_tiles * num_n_tiles
#         # iterate through the tiles in the current gemm problem
#         while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
#             b_ptr = tl.load(weights_ptrs + expert_id).to(tl.pointer_type(tl_calcu_type))
#             tile_idx_in_gemm = tile_idx - last_problem_end
#             tile_m_idx = tile_idx_in_gemm // num_n_tiles
#             tile_n_idx = tile_idx_in_gemm % num_n_tiles

#             # do regular gemm here
#             offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             offs_k = tl.arange(0, BLOCK_SIZE_K)

#             a_index =
#             a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
#             b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
#             accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#             for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
#                 # hint to Triton compiler to do proper loop pipelining
#                 tl.multiple_of(a_ptrs, [16, 16])
#                 tl.multiple_of(b_ptrs, [16, 16])
#                 # assume full tile for now
#                 a = tl.load(a_ptrs)
#                 b = tl.load(b_ptrs)
#                 accumulator += tl.dot(a, b)
#                 a_ptrs += BLOCK_SIZE_K
#                 b_ptrs += BLOCK_SIZE_K * ldb
#             c = accumulator.to(tl.float16)

#             offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#             offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#             c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

#             # assumes full tile for now
#             tl.store(c_ptrs, c)

#             # go to the next tile by advancing NUM_SM
#             tile_idx += NUM_SM

#         # get ready to go to the next gemm problem
#         last_problem_end = last_problem_end + num_tiles


# def group_gemm_fn(group_A, group_B):
#     assert len(group_A) == len(group_B)
#     group_size = len(group_A)

#     A_addrs = []
#     B_addrs = []
#     C_addrs = []
#     g_sizes = []
#     g_lds = []
#     group_C = []
#     for i in range(group_size):
#         A = group_A[i]
#         B = group_B[i]
#         assert A.shape[1] == B.shape[0]
#         M, K = A.shape
#         K, N = B.shape
#         C = torch.empty((M, N), device=DEVICE, dtype=A.dtype)
#         group_C.append(C)
#         A_addrs.append(A.data_ptr())
#         B_addrs.append(B.data_ptr())
#         C_addrs.append(C.data_ptr())
#         g_sizes += [M, N, K]
#         g_lds += [A.stride(0), B.stride(0), C.stride(0)]

#     # note these are device tensors
#     d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
#     d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
#     d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
#     d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
#     d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)
#     # we use a fixed number of CTA, and it's auto-tunable
#     grid = lambda META: (META['NUM_SM'], )
#     grouped_matmul_kernel[grid](
#         d_a_ptrs,
#         d_b_ptrs,
#         d_c_ptrs,
#         d_g_sizes,
#         d_g_lds,
#         group_size,
#     )

#     return group_C


# group_m = [1024, 512, 256, 128]
# group_n = [1024, 512, 256, 128]
# group_k = [1024, 512, 256, 128]
# group_A = []
# group_B = []
# assert len(group_m) == len(group_n)
# assert len(group_n) == len(group_k)
# group_size = len(group_m)
# for i in range(group_size):
#     M = group_m[i]
#     N = group_n[i]
#     K = group_k[i]
#     A = torch.rand((M, K), device=DEVICE, dtype=torch.float16)
#     B = torch.rand((K, N), device=DEVICE, dtype=torch.float16)
#     group_A.append(A)
#     group_B.append(B)

# tri_out = group_gemm_fn(group_A, group_B)
# ref_out = [torch.matmul(a, b) for a, b in zip(group_A, group_B)]
# for i in range(group_size):
#     assert torch.allclose(ref_out[i], tri_out[i], atol=1e-2, rtol=0)


# # only launch the kernel, no tensor preparation here to remove all overhead
# def triton_perf_fn(a_ptrs, b_ptrs, c_ptrs, sizes, lds, group_size):
#     grid = lambda META: (META['NUM_SM'], )
#     grouped_matmul_kernel[grid](
#         a_ptrs,
#         b_ptrs,
#         c_ptrs,
#         sizes,
#         lds,
#         group_size,
#     )


# def torch_perf_fn(group_A, group_B):
#     for a, b in zip(group_A, group_B):
#         torch.matmul(a, b)


# @triton.testing.perf_report(
#     triton.testing.Benchmark(
#         # argument names to use as an x-axis for the plot
#         x_names=['N'],
#         x_vals=[2**i for i in range(7, 11)],  # different possible values for `x_name`
#         line_arg='provider',
#         # argument name whose value corresponds to a different line in the plot
#         # possible values for `line_arg``
#         line_vals=['cublas', 'triton'],
#         # label name for the lines
#         line_names=["cuBLAS", "Triton"],
#         # line styles
#         styles=[('green', '-'), ('blue', '-')],
#         ylabel="runtime(ms)",  # label name for the y-axis
#         plot_name="group-gemm-performance",
#         # name for the plot. Used also as a file name for saving the plot.
#         args={},
#     ))

# def benchmark(N, provider):
#     group_size = 4
#     group_A = []
#     group_B = []
#     A_addrs = []
#     B_addrs = []
#     C_addrs = []
#     g_sizes = []
#     g_lds = []
#     group_C = []
#     for i in range(group_size):
#         A = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
#         B = torch.rand((N, N), device=DEVICE, dtype=torch.float16)
#         C = torch.empty((N, N), device=DEVICE, dtype=torch.float16)
#         group_A.append(A)
#         group_B.append(B)
#         group_C.append(C)
#         A_addrs.append(A.data_ptr())
#         B_addrs.append(B.data_ptr())
#         C_addrs.append(C.data_ptr())
#         g_sizes += [N, N, N]
#         g_lds += [N, N, N]

#     d_a_ptrs = torch.tensor(A_addrs, device=DEVICE)
#     d_b_ptrs = torch.tensor(B_addrs, device=DEVICE)
#     d_c_ptrs = torch.tensor(C_addrs, device=DEVICE)
#     d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=DEVICE)
#     d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=DEVICE)

#     quantiles = [0.5, 0.2, 0.8]
#     if provider == 'cublas':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_perf_fn(group_A, group_B), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(
#             lambda: triton_perf_fn(d_a_ptrs, d_b_ptrs, d_c_ptrs, d_g_sizes, d_g_lds, group_size), quantiles=quantiles)
#     return ms, max_ms, min_ms


# benchmark.run(show_plots=True, print_data=True)
