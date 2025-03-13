"""Fused MoE kernel."""

import torch
import triton
import triton.language as tl
from typing import Any, Callable, Dict, Optional, Tuple
import torch.distributed as dist
from lightllm.utils.log_utils import init_logger
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.common.quantization.triton_quant.fp8.fp8act_quant_kernel import per_token_group_quant_fp8
from lightllm.utils.dist_utils import get_current_device_id
import numpy as np
logger = init_logger(__name__)

import deep_ep
from deep_ep import Buffer, EventOverlap
import deep_gemm

# Set the number of SMs to use
Buffer.set_num_sms(24)
import os
from deep_gemm import bench_kineto, calc_diff, ceil_div, get_col_major_tma_aligned_tensor

def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '8361'))
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    node_rank = int(os.getenv('RANK', 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')
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

def per_token_cast_to_fp8(x: torch.Tensor):
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)

def fused_experts_impl_ref(
    x: torch.Tensor, # [M, K]
    w1: torch.Tensor, # [group, N, K]
    w2: torch.Tensor, # [group, K, N/2]
    topk_weight: torch.Tensor, # [M, topk]
    topk_ids: torch.Tensor, # [M, topk]
    num_experts: int
):
    N = w1.shape[1]
    ep_size = torch.distributed.get_world_size()
    experts_per_rank = num_experts // ep_size

    cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    sorted_tokens_shape = sorted_tokens.shape
    
    if ep_size > 1:
        tokens_per_ep_rank = tokens_per_expert.view(ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(
            tokens_per_expert.shape[0]
        )
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = (
            tokens_per_expert_group.view(ep_size, -1)
            .sum(1)
            .cpu()
            .numpy()
            .tolist()
        )
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(
            ep_size, experts_per_rank
        ).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]

        expert_out = tokens_for_this_expert
        # expert_out = torch.matmul(tokens_for_this_expert, w1[i].T)

        # w1_out = torch.matmul(tokens_for_this_expert, w1[i,:N//2,:].T)
        # w2_out = torch.matmul(tokens_for_this_expert, w1[i,N//2:,:].T)             

        # tmp = w1_out * w2_out
        # tmp = torch.nn.functional.silu(tmp)

        # expert_out = torch.matmul(tmp, w2[i].T)

        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    if ep_size > 1:
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )

    return final_out

def fused_experts_impl(
    hidden_states: torch.Tensor, # [M, K]
    w1: torch.Tensor, # [group, N, K]
    w2: torch.Tensor, # [group, K, N/2]
    topk_weights: torch.Tensor, # [M, topk]
    topk_idx: torch.Tensor, # [M, topk]
    num_experts: int,
    buffer: Buffer,
    prefill: bool,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_fp8_all2all: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None, 
    w2_scale: Optional[torch.Tensor] = None,
    previous_event: Optional[EventOverlap] = None
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

    # qaunt hidden_states
    assert use_fp8_w8a8 and use_fp8_all2all, "use_fp8_w8a8 and use_fp8_all2all must be True"

    block_size_n = 0
    block_size_k = 0

    if w1.ndim == 3:
        block_size_n = w1.shape[1] // w1_scale.shape[1]
        block_size_k = w1.shape[2] // w1_scale.shape[2]
    
    assert block_size_k != 0, "block_size_k can not be 0"

    # input_scale = torch.empty((M, K // block_size_k), dtype=torch.float32, device=hidden_states.device)
    # qinput_tensor = torch.empty((M, K), dtype=w1.dtype, device=hidden_states.device)
    # per_token_group_quant_fp8(hidden_states, block_size_k, qinput_tensor, input_scale)
    qinput_tensor, input_scale = per_token_cast_to_fp8(hidden_states)

    combined_x = None
    if prefill:
        # get_dispatch_layout
        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, previous_event  = \
        buffer.get_dispatch_layout(topk_idx, num_experts, previous_event=previous_event, async_finish=False,
                                        allocate_on_comm_stream=False)

        # normal dispatch
        recv_x, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, handle, event = \
        buffer.dispatch( hidden_states, topk_idx=topk_idx, topk_weights=topk_weights,
                         num_tokens_per_rank=num_tokens_per_rank, num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
                         is_token_in_rank=is_token_in_rank, num_tokens_per_expert=num_tokens_per_expert,
                         previous_event=previous_event, async_finish=False,
                         allocate_on_comm_stream=False)
        if dist.get_rank() == 0:
            print(recv_x.shape, num_recv_tokens_per_expert_list, recv_topk_idx, recv_topk_weights)

        # groupgemm (contiguous layout)
        # gemm_out_a = torch.empty((recv_x[0].shape[0], N), device=hidden_states.device, dtype=hidden_states.dtype)
        # m_indices = torch.empty(recv_x[0].shape[0], device=hidden_states.device, dtype=torch.int)
        # cur = 0
        # for i, num in enumerate(num_recv_tokens_per_expert_list):
        #     m_indices[cur:cur+num] = i
        #     cur += num
        
        # print(recv_x[1].shape, num_recv_tokens_per_expert_list, m_indices)
        # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(recv_x, (w1,w1_scale), gemm_out_a, m_indices)

        # # silu_and_mul_fwd
        # silu_out = torch.empty((recv_x[0].shape[0], N//2), device=hidden_states.device, dtype=hidden_states.dtype)
        # # silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)
        # silu_out = torch.nn.functional.silu(gemm_out_a[:, :N//2] * gemm_out_a[:, N//2:])

        # # groupgemm (contiguous layout)
        # # qsilu_out_scale = torch.empty((recv_x[0].shape[0], N//2//128), dtype=torch.float32, device=hidden_states.device)
        # # qsilu_out = torch.empty((recv_x[0].shape[0], N//2), dtype=w1.dtype, device=hidden_states.device)
        # qsilu_out, qsilu_out_scale = per_token_cast_to_fp8(silu_out)
        # # per_token_group_quant_fp8(silu_out, block_size_k, qsilu_out, qsilu_out_scale)

        # gemm_out_b = torch.empty_like(recv_x[0], device=hidden_states.device, dtype=hidden_states.dtype)
        # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((qsilu_out,qsilu_out_scale), (w2,w2_scale), gemm_out_b, m_indices)

        # normal combine
        combined_x, recv_topk_weights, event = buffer.combine(recv_x, handle, topk_weights=recv_topk_weights, async_finish=False, previous_event=previous_event,
                                           allocate_on_comm_stream=False)
        # if dist.get_rank() == 0:
        #     print(combined_x.shape, recv_topk_weights, is_token_in_rank, is_token_in_rank.sum(dim=1).unsqueeze(1))
        combined_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)

    else:
        num_max_dispatch_tokens_per_rank = M

        # low latency dispatch
        recv_x, recv_expert_count, handle, event, hook = \
        buffer.low_latency_dispatch(hidden_states, topk_idx, num_max_dispatch_tokens_per_rank, num_experts, use_fp8=False,
                                     async_finish=False, return_recv_hook=True)
        hook()

        # # groupgemm (masked layout)
        # gemm_out_a = torch.empty((recv_x[0].shape[0], N), device=hidden_states.device, dtype=hidden_states.dtype)
        # masked_m = torch.empty((E, ), device='cuda', dtype=torch.int)
        # for j in range(E):
        #     masked_m[j] = recv_expert_count[j]
        # expected_m = min(int(masked_m.float().mean()) + 1, num_max_dispatch_tokens_per_rank)
        # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(recv_x, (w1,w1_scale), gemm_out_a, masked_m, expected_m)

        # # silu_and_mul_fwd
        # silu_out = torch.empty((recv_x[0].shape[0], N//2), device=hidden_states.device, dtype=hidden_states.dtype)
        # silu_and_mul_fwd(gemm_out_a.view(-1, N), silu_out)

        # # groupgemm (masked layout)
        # qsilu_out_scale = torch.empty((recv_x[0].shape[0], N//2//128), dtype=torch.float32, device=hidden_states.device)
        # qsilu_out = torch.empty((recv_x[0].shape[0], N//2), dtype=w1.dtype, device=hidden_states.device)
        # per_token_group_quant_fp8(hidden_states, block_size_k, qsilu_out, qsilu_out_scale)

        # print(hidden_states.shape, qsilu_out.shape, qsilu_out_scale.shape)
        # gemm_out_b = torch.empty_like(recv_x[0], device=hidden_states.device, dtype=hidden_states.dtype)
        # deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked((qsilu_out,qsilu_out_scale), (w2,w2_scale), gemm_out_b, masked_m, expected_m)

        # low latency combine
        combined_x, event_overlap, hook = \
        buffer.low_latency_combine(recv_x, topk_idx, topk_weights, handle,
                                    async_finish=False, return_recv_hook=True)
        hook()
        # combined_x = combined_x.float() / is_token_in_rank.sum(dim=1).unsqueeze(1)

    return combined_x

def test_loop(local_rank: int, num_local_ranks: int):
    # 初始化 dist
    num_nodes = int(os.getenv('WORLD_SIZE', 1))
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(rank)

    # 构造fused_experts_impl的输入参数
    seqlen = 1
    hidden_states = torch.randn((seqlen, 7168), device='cuda', dtype=torch.bfloat16) 
    w1 = torch.randn((256//num_local_ranks, 7168, 7168), device='cuda', dtype=torch.bfloat16)  
    w2 = torch.randn((256//num_local_ranks, 7168, 2048), device='cuda', dtype=torch.bfloat16)
    
    w1_fp8 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((256//num_local_ranks, 7168//128, 7168//128), device='cuda', dtype=torch.float)
    w2_scale = torch.empty((256//num_local_ranks, 7168//128, 2048//128), device='cuda', dtype=torch.float)

    for i in range(256//num_local_ranks):
        w1_fp8[i], w1_scale[i] = per_block_cast_to_fp8(w1[i])
        w2_fp8[i], w2_scale[i] = per_block_cast_to_fp8(w2[i])

    topk_weights = torch.randn((seqlen, 8), device='cuda', dtype=torch.float32)
    topk_weights = torch.softmax(topk_weights, dim=-1)  # 对每行进行softmax归一化
    topk_ids = torch.zeros((seqlen, 8), device='cuda', dtype=torch.int64)
    for i in range(seqlen):
        topk_ids[i] = torch.randperm(254, device='cuda')[:8] + 1

    # 初始化 buffer
    test_ll_compatibility, num_rdma_bytes = True, 0
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, ll_num_topk = 256, 7168, 256, 8
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(ll_num_tokens, ll_hidden, num_ranks, ll_num_experts)

    buffer = deep_ep.Buffer(group, int(1e9), num_rdma_bytes, low_latency_mode=test_ll_compatibility,
                            num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1))

    # 调用fused_experts_impl
    ref_output = fused_experts_impl_ref(x=hidden_states, w1=w1, w2=w2, topk_weight=topk_weights ,topk_ids=topk_ids, num_experts=256)

    output = fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1_fp8,
        w2=w2_fp8,
        topk_weights=topk_weights,
        topk_idx=topk_ids,
        num_experts=256,
        buffer=buffer,
        prefill=True,
        inplace=False,
        use_fp8_w8a8=True, 
        use_fp8_all2all=True,
        use_int8_w8a16=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        previous_event=None
    )

    # Test compatibility with low latency functions
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        ll_output = fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1_fp8,
            w2=w2_fp8,
            topk_weights=topk_weights,
            topk_idx=topk_ids,
            num_experts=256,
            buffer=buffer,
            prefill=False,
            inplace=False,
            use_fp8_w8a8=True, 
            use_fp8_all2all=True,
            use_int8_w8a16=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            previous_event=None
        ) 

    if dist.get_rank() == 0:
        print(ref_output, output, ll_output)

    dist.barrier()

if __name__ == '__main__':
    num_processes = 8
    torch.multiprocessing.spawn(test_loop, args=(num_processes, ), nprocs=num_processes)