import time
import torch
import itertools
from typing import Optional, Tuple
from vllm import _custom_ops as ops
from sgl_kernel import sgl_per_token_quant_fp8

try:
    from lightllm_kernel.ops import per_token_quant_bf16_fp8
except ImportError:
    raise ImportError("lightllm-kernel op per_token_quant_bf16_fp8 not found.")

fp8_type_ = torch.float8_e4m3fn


def vllm_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ops.scaled_fp8_quant(input, use_per_token_if_dynamic=True)


def sglang_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
    output = torch.empty_like(input, device=input.device, dtype=fp8_type_)
    sgl_per_token_quant_fp8(input, output, scale)

    return output, scale


def lightllm_per_token_quant_fp8(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return per_token_quant_bf16_fp8(input)


def dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.to(torch.bfloat16) * scale.view(-1, *((1,) * (q.dim() - 1)))


def benchmark(fn, name, inp, iterations=200):
    for _ in range(20):
        q, s = fn(inp)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iterations):
        q, s = fn(inp)
    ender.record()
    torch.cuda.synchronize()
    avg_ms = starter.elapsed_time(ender) / iterations

    q, s = fn(inp)
    recon = dequantize(q, s)
    err = recon - inp.to(torch.bfloat16)
    mse = err.pow(2).mean().item()
    max_err = err.abs().max().item()

    print(f"{name:20s} | latency: {avg_ms:7.3f} ms | MSE: {mse:.3e} | MaxErr: {max_err:.3e}")


if __name__ == "__main__":
    batch, seq_len = 64, 4096
    device = "cuda"
    inp = torch.randn(batch, seq_len, device=device, dtype=torch.bfloat16)

    benchmark(vllm_per_token_quant_fp8, "vllm_ops", inp)
    benchmark(sglang_per_token_quant_fp8, "sgl_kernel", inp)
    benchmark(lightllm_per_token_quant_fp8, "lightllm_kernel", inp)
