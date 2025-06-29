import torch
from typing import Optional, Tuple
from . import _C


def per_token_quant_bf16_fp8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    _C.per_token_quant_bf16_fp8(output, input, scales)
    return output, scales

def per_token_quant_bf16_int8(input: torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize the given input using per token quant method"""
    output = torch.empty_like(input, dtype=torch.int8)
    scales = torch.empty(size=(input.shape[0], 1), device=input.device, dtype=torch.float32)
    _C.per_token_quant_bf16_int8(output, input, scales)
    return output, scales
