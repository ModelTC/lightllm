"""Fused MoE kernel."""

from typing import Any, Callable, Dict, Optional, Tuple
import torch
import triton
import triton.language as tl
from lightllm.utils.log_utils import init_logger

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


@triton.jit
def grouped_matmul_kernel(
    k,  # int
    n,  # int
    expert_num,  # int
    topk_num,  # int
    token_ptr,  # [token_num, hidden_dim]
    token_stride_0,
    token_stride_1,
    weights_ptr,  # [expert_num, N, K]
    weight_stride_0,
    weight_stride_1,
    weight_stride_2,
    expert_to_token_num,  # [expert_num]
    expert_to_token_index,  # [expert_num, token_num * topk_num]
    expert_to_token_index_stride_0,
    out_ptr,  # [token_num * topk_num, n]
    out_stride_0,
    out_stride_1,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    last_problem_end = 0

    out_dtype = token_ptr.dtype.element_ty
    for expert_id in range(expert_num):
        # get the gemm size of the current problem
        cur_m = tl.load(expert_to_token_num + expert_id)
        num_m_tiles = tl.cdiv(cur_m, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(n, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:

            tile_idx_in_gemm = tile_idx - last_problem_end

            # better super-grouping for L2 Cache Optimizations
            pid = tile_idx_in_gemm
            num_pid_m = num_m_tiles
            num_pid_n = num_n_tiles
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            group_id = pid // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            in_group_index = pid % num_pid_in_group
            back_mark = (in_group_index // group_size_m) % 2
            back_mark1 = -1 * (2 * back_mark - 1)
            pid_m = first_pid_m + back_mark * (group_size_m - 1) + back_mark1 * (in_group_index % group_size_m)
            pid_n = (pid % num_pid_in_group) // group_size_m

            tile_m_idx = pid_m
            tile_n_idx = pid_n

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            a_m_index = tl.load(
                expert_to_token_index + expert_id * expert_to_token_index_stride_0 + offs_am,
                mask=offs_am < cur_m,
                other=0,
            )

            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            a_ptrs = token_ptr + (a_m_index // topk_num)[:, None] * token_stride_0 + offs_k[None, :]
            b_ptrs = weights_ptr + weight_stride_0 * expert_id + offs_k[:, None] + offs_bn[None, :] * weight_stride_1
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for _ in range(0, tl.cdiv(k, BLOCK_SIZE_K)):
                # hint to Triton compiler to do proper loop pipelining
                # tl.multiple_of(a_ptrs, [16, 16])
                tl.multiple_of(b_ptrs, [16, 16])
                a = tl.load(a_ptrs, mask=(offs_am[:, None] < cur_m) & (offs_k[None, :] < k))
                b = tl.load(b_ptrs, mask=(offs_bn[None, :] < n) & (offs_k[:, None] < k))
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K
                offs_k += BLOCK_SIZE_K

            c = accumulator.to(out_dtype)

            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = out_ptr + a_m_index[:, None] * out_stride_0 + offs_cn[None, :]
            tl.store(c_ptrs, c, mask=(offs_am[:, None] < cur_m) & (offs_cn[None, :] < n))
            tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles
    return


def grouped_matmul(
    token_inputs: torch.Tensor,
    expert_to_token_num: torch.Tensor,
    expert_to_token_index: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_num: int,
    out: torch.Tensor,
):
    """
    token_inputs is tensor shape [token_num, hidden_dim],
    expert_to_token_num is tensor shape [expert_num],
    expert_to_token_index is tensor shape [expert_num, token_num * topk_num],
    expert_weights is tensor shape [expert_num, out_dim, hidden_dim]
    out is tensor shape [token_num * topk_num, out_dim]
    """

    NUM_SM = 128
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 1
    num_stages = 8
    num_warps = 4

    expert_num, n, k = expert_weights.shape
    assert token_inputs.shape[1] == k

    grid = (NUM_SM,)

    grouped_matmul_kernel[grid](
        k,
        n,
        expert_num,
        topk_num,
        token_inputs,
        token_inputs.stride(0),
        token_inputs.stride(1),
        expert_weights,
        expert_weights.stride(0),
        expert_weights.stride(1),
        expert_weights.stride(2),
        expert_to_token_num,
        expert_to_token_index,
        expert_to_token_index.stride(0),
        out,
        out.stride(0),
        out.stride(1),
        NUM_SM=NUM_SM,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return


# def fused_experts_impl(
#     hidden_states: torch.Tensor,
#     w1: torch.Tensor,
#     w2: torch.Tensor,
#     topk_weights: torch.Tensor,
#     topk_ids: torch.Tensor,
#     inplace: bool = False,
#     use_fp8_w8a8: bool = False,
#     use_int8_w8a16: bool = False,
#     w1_scale: Optional[torch.Tensor] = None,
#     w2_scale: Optional[torch.Tensor] = None,
#     a1_scale: Optional[torch.Tensor] = None,
#     a2_scale: Optional[torch.Tensor] = None,
#     alloc_tensor_func=torch.empty,
# ):
#     # Check constraints.
#     assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
#     assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
#     assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
#     assert w1.is_contiguous(), "Expert weights1 must be contiguous"
#     assert w2.is_contiguous(), "Expert weights2 must be contiguous"
#     assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

#     num_tokens, _ = hidden_states.shape
#     E, N, _ = w1.shape
#     # We execute the fused_moe kernel in chunks to circumvent this issue:
#     # https://github.com/vllm-project/vllm/issues/5938
#     CHUNK_SIZE = 8 * 1024
#     M = min(num_tokens, CHUNK_SIZE)

#     intermediate_cache1 = alloc_tensor_func(
#         (M, topk_ids.shape[1], N), device=hidden_states.device, dtype=hidden_states.dtype
#     )
#     intermediate_cache2 = alloc_tensor_func(
#         (M * topk_ids.shape[1], N // 2), device=hidden_states.device, dtype=hidden_states.dtype
#     )
#     intermediate_cache3 = alloc_tensor_func(
#         (M, topk_ids.shape[1], w2.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype
#     )

#     compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

#     if inplace:
#         out_hidden_states = hidden_states
#     else:
#         out_hidden_states = alloc_tensor_func(
#             hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype
#         )

#     for chunk in range((num_tokens // CHUNK_SIZE) + 1):
#         begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE, min((chunk + 1) * CHUNK_SIZE, num_tokens))
#         curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
#         tokens_in_chunk, _ = curr_hidden_states.shape

#         if tokens_in_chunk == 0:
#             break

#         if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
#             # Adjust the intermediate cache size and config for the last
#             # chunk. Note that in most cases we only have one chunk
#             # so the cache size and config are already set correctly and
#             # do not need to be adjusted.
#             intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
#             intermediate_cache2 = intermediate_cache2[:tokens_in_chunk]
#             intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
#             config = get_config_func(tokens_in_chunk)

#         curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
#         curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

#         # 注意 moe_align_block_size 函数不能使用框架自己的缓存tensor管理框架
#         # 主要是其申请的tensor大小没有与batch size的线性关系，导致与缓存tensor管理
#         # 框架存在了一些不兼容的情况
#         sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
#             curr_topk_ids, config["BLOCK_SIZE_M"], E, alloc_tensor_func
#         )

#         invoke_fused_moe_kernel(
#             curr_hidden_states,
#             w1,
#             intermediate_cache1,
#             a1_scale,
#             w1_scale,
#             curr_topk_weights,
#             curr_topk_ids,
#             sorted_token_ids,
#             expert_ids,
#             num_tokens_post_padded,
#             False,
#             topk_ids.shape[1],
#             config,
#             compute_type=compute_type,
#             use_fp8_w8a8=use_fp8_w8a8,
#             use_int8_w8a16=use_int8_w8a16,
#         )

#         ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

#         invoke_fused_moe_kernel(
#             intermediate_cache2,
#             w2,
#             intermediate_cache3,
#             a2_scale,
#             w2_scale,
#             curr_topk_weights,
#             curr_topk_ids,
#             sorted_token_ids,
#             expert_ids,
#             num_tokens_post_padded,
#             True,
#             1,
#             config,
#             compute_type=compute_type,
#             use_fp8_w8a8=use_fp8_w8a8,
#             use_int8_w8a16=use_int8_w8a16,
#         )

#         ops.moe_sum(
#             intermediate_cache3.view(*intermediate_cache3.shape), out_hidden_states[begin_chunk_idx:end_chunk_idx]
#         )
#     return out_hidden_states
