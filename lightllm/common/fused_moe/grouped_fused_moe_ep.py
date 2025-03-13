"""Fused MoE kernel."""
# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/fused_moe/fused_moe.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, Optional, Tuple
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.dist_utils import get_current_device_id

FFN_MOE_CHUNK_SIZE = 8 * 1024

logger = init_logger(__name__)

import deep_ep
from deep_ep import Buffer, EventOverlap
import deep_gemm

# # Communication buffer (will allocate at runtime)
# _buffer: Optional[Buffer] = None

# Set the number of SMs to use
Buffer.set_num_sms(24)


def fused_experts_impl(
    hidden_states: torch.Tensor,  # [M, K]
    w1: torch.Tensor,  # [group, N, K]
    w2: torch.Tensor,  # [group, K, N/2]
    topk_weights: torch.Tensor,  # [M, topk]
    topk_idx: torch.Tensor,  # [M, topk]
    num_experts: int,
    _buffer: Buffer,
    prefill: bool,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_fp8_all2all: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    previous_event: Optional[EventOverlap] = None,
):
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_idx.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    M, K = hidden_states.shape
    E, N, _ = w1.shape

    # qaunt hidden_states (in:[M,H] out:([M,H],[M,H/128]) )
    assert use_fp8_w8a8 and use_fp8_all2all, "use_fp8_w8a8 and use_fp8_all2all must be True"

    block_size_k = 0

    if w1.ndim == 3:
        # block_size_n = w1.shape[1] // w1_scale.shape[1]
        block_size_k = w1.shape[2] // w1_scale.shape[2]

    assert block_size_k != 0, "block_size_k can not be 0"

    input_scale = torch.empty((M, K // block_size_k), dtype=torch.float32, device=hidden_states.device)
    qinput_tensor = torch.empty((M, K), dtype=w1.dtype, device=hidden_states.device)
    per_token_group_quant_fp8(hidden_states, block_size_k, qinput_tensor, input_scale)

    # get_dispatch_layout
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = _buffer.get_dispatch_layout(
        topk_idx, num_experts, previous_event=previous_event, async_finish=False, allocate_on_comm_stream=False
    )

    if prefill:
        # normal dispatch
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = _buffer.dispatch(
            (qinput_tensor, input_scale),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        # groupgemm (contiguous layout)
        gemm_out_a = torch.empty((recv_x[0].shape[0], N), device=hidden_states.device, dtype=hidden_states.dtype)
        m_indices = torch.empty(recv_x[0].shape[0], device=hidden_states.device, dtype=torch.int)

        # todo
        cur = 0
        for i, num in enumerate(num_recv_tokens_per_expert_list):
            m_indices[cur:num] = i
            cur += num

        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(recv_x, (w1, w1_scale), gemm_out_a, m_indices)

        # silu_and_mul_fwd
        silu_out = torch.empty((recv_x[0].shape[0], N // 2), device=hidden_states.device, dtype=hidden_states.dtype)
        silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)

        # groupgemm (contiguous layout)
        qsilu_out_scale = torch.empty(
            (recv_x[0].shape[0], N // 2 // 128), dtype=torch.float32, device=hidden_states.device
        )
        qsilu_out = torch.empty((recv_x[0].shape[0], N // 2), dtype=w1.dtype, device=hidden_states.device)
        per_token_group_quant_fp8(hidden_states, block_size_k, qsilu_out, qsilu_out_scale)

        gemm_out_b = torch.empty_like(recv_x[0], device=hidden_states.device, dtype=hidden_states.dtype)
        deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
            (qsilu_out, qsilu_out_scale), (w2, w2_scale), gemm_out_b, m_indices
        )

        # normal combine
        combined_x, _, event = _buffer.combine(
            gemm_out_b, handle, async_finish=False, previous_event=previous_event, allocate_on_comm_stream=False
        )

        hidden_states[:] = combined_x

    # else:
    # low latency dispatch

    # groupgemm (masked layout)

    # silu_and_mul_fwd

    # groupgemm (masked layout)

    # low latency combine

    return combined_x


import os
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "10.120.114.75")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def test_loop(local_rank: int, num_local_ranks: int):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    test_ll_compatibility, num_rdma_bytes = False, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, _ = 16, 5120, 256, 9
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            ll_num_tokens, ll_hidden, num_ranks, ll_num_experts
        )

    buffer = deep_ep.Buffer(
        group,
        int(1e9),
        num_rdma_bytes,
        low_latency_mode=test_ll_compatibility,
        num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),
    )
    torch.manual_seed(rank)

    # 构造fused_experts_impl的输入参数
    hidden_states = torch.randn((4096, 7168), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn((256 // num_local_ranks, 4096, 7168), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((256 // num_local_ranks, 7168, 2048), device="cuda", dtype=torch.bfloat16)

    w1_fp8 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((256 // num_local_ranks, 4096 // 128, 7168 // 128), device="cuda", dtype=torch.float)
    w2_scale = torch.empty((256 // num_local_ranks, 7168 // 128, 2048 // 128), device="cuda", dtype=torch.float)

    for i in range(256 // num_local_ranks):
        w1_fp8[i], w1_scale[i] = per_block_cast_to_fp8(w1[i])
        w2_fp8[i], w2_scale[i] = per_block_cast_to_fp8(w2[i])

    topk_weights = torch.randn((4096, 8), device="cuda", dtype=torch.float32)  # topk权重
    topk_ids = torch.randint(0, 256, (4096, 8), device="cuda", dtype=torch.int64)  # topk索引

    # 调用fused_experts_impl
    fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1_fp8,
        w2=w2_fp8,
        topk_weights=topk_weights,
        topk_idx=topk_ids,
        num_experts=256,
        _buffer=buffer,
        prefill=not test_ll_compatibility,
        inplace=False,
        use_fp8_w8a8=True,  # 启用FP8
        use_fp8_all2all=True,
        use_int8_w8a16=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        previous_event=None,
    )

    # # Test compatibility with low latency functions
    # if test_ll_compatibility:
    #     buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
    #     test_low_latency.test_main(
    #         ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk, rank, num_ranks, group, buffer, seed=1
    #     )


if __name__ == "__main__":
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes,), nprocs=num_processes)
