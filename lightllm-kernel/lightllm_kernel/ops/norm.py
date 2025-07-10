import torch
from typing import Optional
from . import _C


def rmsnorm_bf16(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return _C.rmsnorm_align16_bf16(X, W, eps)
