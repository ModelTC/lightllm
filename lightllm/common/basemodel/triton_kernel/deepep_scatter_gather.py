import random
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Dict


@triton.jit
def _fwd_kernel_ep_scatter(
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
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_expert = tl.program_id(0)
    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    off_d = tl.arange(0, BLOCK_D)

    for start_m in range(0, cur_expert_token_num, BLOCK_E):
        tl.store(
            m_indices_start_ptr + start_m + off_expert,
            cur_expert,
        )

    output_tensor_ptr = output_tensor + cur_expert_start * output_tensor_stride0
    output_tensor_scale_ptr = output_tensor_scale + cur_expert_start * output_tensor_scale_stride0
    cur_index = 0
    for start_topk in range(0, topk_row * topk_col):
        topk = tl.load(recv_topk + start_topk)
        if topk == cur_expert:
            recv_x_index = start_topk // topk_col
            for start_d in range(0, HIDDEN_SIZE, BLOCK_D):
                tl.store(
                    output_tensor_ptr + start_d + off_d,
                    tl.load(recv_x + recv_x_index * recv_x_stride0 + start_d + off_d),
                )
                tl.store(
                    output_tensor_scale_ptr + start_d // BLOCK_D,
                    tl.load(recv_x_scale + recv_x_index * recv_x_scale_stride0 + start_d // BLOCK_D),
                )
            tl.store(output_index + start_topk, cur_index)
            cur_index += 1
            output_tensor_ptr += output_tensor_stride0
            output_tensor_scale_ptr += output_tensor_scale_stride0


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
    num_warps = 4
    num_experts = expert_start_loc.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts
    _fwd_kernel_ep_scatter[(grid,)](
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
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
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
    BLOCK_D: tl.constexpr,
):
    cur_block = tl.program_id(0)
    cur_token = tl.program_id(1)
    off_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start_topk in range(0, topk_col):
        expert_id = tl.load(recv_topk + cur_token * recv_topk_stride0 + start_topk)
        if expert_id >= 0:
            expert_offset = tl.load(input_index + cur_token * input_index_stride0 + start_topk)
            cur_expert_start = tl.load(expert_start_loc + expert_id)
            acc_weight = tl.load(recv_topk_weight + cur_token * recv_topk_weight_stride0 + start_topk)
            tmp = tl.load(
                input_tensor + (cur_expert_start + expert_offset) * input_tensor_stride0 + cur_block * BLOCK_D + off_d
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
    num_warps = 4
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 65535))
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
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


def test_case1():
    block_size = 128
    num_recv_tokens_per_expert_list = [0] * 32
    num_recv_tokens_per_expert_list[6] = 128
    num_recv_tokens_per_expert_list[7] = 128
    num_recv_tokens_per_expert_list[8] = 128
    num_recv_tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, dtype=torch.int, device="cuda")

    all_tokens = sum(num_recv_tokens_per_expert_list)
    m_indices_ref = torch.empty(all_tokens, device="cuda", dtype=torch.int32)
    m_indices = torch.empty(all_tokens, device="cuda", dtype=torch.int32)

    recv_x = torch.randn((7, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    recv_x_scale = torch.randn((7, 4096 // block_size), device="cuda", dtype=torch.float32)

    recv_topk_id = torch.ones((7, 8), device="cuda", dtype=torch.int32) * -1
    recv_topk_weights = torch.zeros((7, 8), device="cuda", dtype=torch.float)
    for i in range(7):
        for j in range(4):
            idx = random.randint(0, 7)
            expert_id = random.randint(6, 8)
            recv_topk_id[i][idx] = expert_id
            recv_topk_weights[i][idx] = random.randint(0, 10) / 10.0

    output_indexs = torch.zeros_like(recv_topk_id)
    output_tensor = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    output_tensor_ref = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

    output_tensor_scale = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)
    output_tensor_scale_ref = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)

    expert_start_loc = torch.cumsum(torch.tensor([0] + num_recv_tokens_per_expert_list[:-1], device="cuda"), dim=0)
    cur_pos = torch.zeros_like(expert_start_loc, device="cuda", dtype=torch.int32)

    cur = 0
    for i, k in enumerate(num_recv_tokens_per_expert_list):
        m_indices_ref[cur : cur + k] = i
        cur += k

    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk_id,
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_indexs,
    )
    diff = (m_indices - m_indices_ref).min()
    print(f"m_indices diff: {diff}")
    print(output_indexs)

    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = expert_start_loc[recv_topk_id[i][j]] + cur_pos[recv_topk_id[i][j]]
                output_tensor_ref[dst][:] = recv_x[i][:]
                output_tensor_scale_ref[dst][:] = recv_x_scale[i][:]
                cur_pos[recv_topk_id[i][j]] += 1

    print("output diff tensor: ", (output_tensor.to(torch.float) - output_tensor_ref.to(torch.float)).abs().max())
    print("output scale diff tensor scale: ", (output_tensor_scale - output_tensor_scale_ref).abs().max())

    #### gather

    gather_out_ref = torch.zeros_like(recv_x, device="cuda", dtype=torch.bfloat16)
    gather_out = torch.empty_like(recv_x, device="cuda", dtype=torch.bfloat16)
    cur_pos = torch.zeros_like(expert_start_loc, device="cuda", dtype=torch.int32)
    gather_input = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.bfloat16)
    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = expert_start_loc[recv_topk_id[i][j]] + cur_pos[recv_topk_id[i][j]]
                gather_out_ref[i][:] += gather_input[dst][:] * recv_topk_weights[i][j]
                cur_pos[recv_topk_id[i][j]] += 1
    print(recv_topk_id)
    ep_gather(gather_input, recv_topk_id, recv_topk_weights, output_indexs, expert_start_loc, gather_out)
    print("gather output diff tensor: ", (gather_out - gather_out_ref).abs().max())

    pass


if __name__ == "__main__":
    test_case1()
