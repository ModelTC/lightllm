import torch
import triton
import triton.language as tl


@triton.jit
def softmax_topk_kernel(
    topk_weights_ptr,
    topk_indices_ptr,
    gating_output_ptr,
    input_row_stride,
    output_weights_row_stride,
    output_indices_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    top_k: tl.constexpr,
):
    row_idx = tl.program_id(0)

    row_input_ptr = gating_output_ptr + row_idx * input_row_stride
    row_weights_ptr = topk_weights_ptr + row_idx * output_weights_row_stride
    row_indices_ptr = topk_indices_ptr + row_idx * output_indices_row_stride

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    values = tl.load(row_input_ptr + offsets, mask=mask, other=-float("inf"))

    current_max = tl.max(values, axis=0)
    values = values - current_max
    numerators = tl.exp(values)
    denom = tl.sum(numerators, axis=0)

    for i in range(top_k):
        logit = tl.max(values, axis=0)
        idx = tl.argmax(values, axis=0)

        prob = tl.exp(logit) / denom

        ptr_w = row_weights_ptr + i
        ptr_i = row_indices_ptr + i

        tl.store(ptr_w, prob)
        tl.store(ptr_i, idx)

        values = tl.where(offsets == idx, -float("inf"), values)


def softmax_topk(gating_output: torch.Tensor, topk: int, renorm: bool = False):
    assert gating_output.dim() == 2, "The dim of gating_output must be 2."
    num_tokens, num_experts = gating_output.shape
    device = gating_output.device

    if gating_output.dtype != torch.float32:
        gating_output = gating_output.to(torch.float32)

    topk_vals = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_idxs = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)

    BLOCK_SIZE = triton.next_power_of_2(num_experts)

    grid = (num_tokens,)
    softmax_topk_kernel[grid](
        topk_vals,
        topk_idxs,
        gating_output,
        gating_output.stride(0),
        topk_vals.stride(0),
        topk_idxs.stride(0),
        num_tokens,
        num_experts,
        BLOCK_SIZE=BLOCK_SIZE,
        top_k=topk,
        num_warps=8,
    )
    if renorm:
        row_sum = topk_vals.sum(-1, keepdim=True).clamp_min(1e-8)
        topk_vals.div_(row_sum)
    return topk_vals, topk_idxs


import sgl_kernel as sgl_ops


#
def benchmark(M, N, K, renorm, runs):
    gating = torch.randn(M, N, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()

    # 1. SGL kernel
    sgl_vals = torch.empty((M, K), dtype=torch.float32, device="cuda")
    sgl_ids = torch.empty((M, K), dtype=torch.int32, device="cuda")
    # Warm-up
    sgl_ops.topk_softmax(sgl_vals, sgl_ids, torch.empty_like(sgl_ids), gating)
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        sgl_ops.topk_softmax(sgl_vals, sgl_ids, torch.empty_like(sgl_ids), gating)
    end.record()
    torch.cuda.synchronize()
    t_sgl = start.elapsed_time(end) / runs

    # 2. Triton kernel
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    # Warm-up
    softmax_topk(gating, K)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(runs):
        triton_vals, triton_ids = softmax_topk(gating, K, renorm)
    t1.record()
    torch.cuda.synchronize()
    t_triton = t0.elapsed_time(t1) / runs

    # 3. Native PyTorch
    _ = torch.softmax(gating, dim=-1)
    _, _ = torch.topk(_, K, dim=-1)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(runs):
        probs = torch.softmax(gating, dim=-1)
        torch_vals, torch_ids = torch.topk(probs, K, dim=-1)
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end) / runs

    # Compare indices and weights
    # Count mismatches of ordered indices
    diff_sgl_triton_ids = (sgl_ids != triton_ids).sum().item()
    diff_torch_triton_ids = (torch_ids != triton_ids).sum().item()
    # Max absolute difference of weights aligned by position
    max_err_triton_torch = (triton_vals - torch_vals).abs().max().item()
    max_err_triton_torch_sgl = (sgl_vals - torch_vals).abs().max().item()
    max_err_triton_sgl = (triton_vals - sgl_vals).abs().max().item()

    results = {
        "time_sgl": t_sgl,
        "time_triton": t_triton,
        "time_torch": t_torch,
        "mismatch_sgl_triton_ids": diff_sgl_triton_ids,
        "mismatch_torch_triton_ids": diff_torch_triton_ids,
        "max_err_triton_torch": max_err_triton_torch,
        "max_err_triton_sgl": max_err_triton_sgl,
        "max_err_triton_torch_sgl": max_err_triton_torch_sgl,
        "sgl_ids": sgl_ids,
        "triton_ids": triton_ids,
        "torch_ids": torch_ids,
        "sgl_vals": sgl_vals,
        "triton_vals": triton_vals,
        "torch_vals": torch_vals,
    }
    return results


if __name__ == "__main__":
    # Example: 8192 tokens, 1024 experts, Top-4
    M, N, K = 8192, 1024, 4
    res = benchmark(M, N, K, False, 1000)
    print(f"SGL     time: {res['time_sgl']:.6f}ms")
    print(f"Triton  time: {res['time_triton']:.6f}ms")
    print(f"PyTorch time: {res['time_torch']:.6f}ms")
    print("Mismatch SGL vs Triton ids:", res["mismatch_sgl_triton_ids"])
    print("Mismatch Torch vs Triton ids:", res["mismatch_torch_triton_ids"])
    print("Max err Triton vs Torch  :", res["max_err_triton_torch"])
    print("Max err Triton vs SGL    :", res["max_err_triton_sgl"])
    print("Max err Torch vs SGL    :", res["max_err_triton_torch_sgl"])
