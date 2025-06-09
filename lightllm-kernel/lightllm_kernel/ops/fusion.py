import torch
from typing import Optional, Tuple
from . import _C


def pre_tp_norm_bf16(input: torch.Tensor) -> torch.Tensor:
    """Calculate powersum along embedding dimension of the input"""
    return _C.pre_tp_norm_bf16(input)


def post_tp_norm_bf16(
    input: torch.tensor, weight: torch.Tensor, tp_variance: torch.Tensor, embed_dim: int, eps: float
) -> torch.Tensor:
    """Apply rmsnorm on given input, with weight and pre calculated powersum"""
    return _C.post_tp_norm_bf16(input, weight, tp_variance, embed_dim, eps)


def add_norm_quant_bf16_fp8(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply add_norm_quant on given input, with residual and weight"""
    return _C.add_norm_quant_bf16_fp8(input, residual, weight, eps)


def gelu_per_token_quant_bf16_fp8(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply gelu on given input and quantize it from bf16 to fp8 using per token quant method"""
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    _C.gelu_per_token_quant_bf16_fp8(output, input, scales)
    return output, scales
