import time
import torch
from typing import Optional, Tuple, Union

from vllm import _custom_ops as vllm_ops
from lightllm_kernel.ops import rmsnorm_bf16 as lightllm_rms_norm
from lightllm.models.vit.triton_kernel.rms_norm_vit import rms_norm as triton_rms_norm


def vllm_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
):
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    if residual is not None:
        residual = residual.view(-1, residual.shape[-1])

    if residual is not None:
        vllm_ops.fused_add_rms_norm(x, residual, weight, eps)
        output = (x, residual)
    else:
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, weight, eps)
        output = out

    if isinstance(output, tuple):
        output = (output[0].view(orig_shape), output[1].view(orig_shape))
    else:
        output = output.view(orig_shape)
    return output


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float):
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(mean_sq + eps)
    out = x * inv_std * w
    return out


def benchmark(fn, name, x, w, eps, iterations=200):
    for _ in range(10):
        _ = fn(x, w, eps)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iterations):
        _ = fn(x, w, eps)
    ender.record()
    torch.cuda.synchronize()
    latency_ms = starter.elapsed_time(ender) / iterations

    y_ref = torch_rmsnorm(x, w, eps)
    y_out = fn(x, w, eps)
    err = y_out - y_ref
    mse = err.pow(2).mean().item()
    max_err = err.abs().max().item()

    print(f"{name:20s} | latency: {latency_ms:7.3f} ms | MSE: {mse:.3e} | MaxErr: {max_err:.3e}")


if __name__ == "__main__":

    batch, dim = 64, 1024
    eps = 1e-6
    device = "cuda"

    x = torch.randn(batch, dim, device=device, dtype=torch.bfloat16)
    w = torch.randn(dim, device=device, dtype=torch.bfloat16)

    benchmark(torch_rmsnorm, "torch_rmsnorm", x, w, eps)
    benchmark(lightllm_rms_norm, "lightllm_rms_norm", x, w, eps)
    benchmark(triton_rms_norm, "triton_rms_norm", x, w, eps)
    benchmark(vllm_rmsnorm, "vllm_rmsnorm", x, w, eps)
