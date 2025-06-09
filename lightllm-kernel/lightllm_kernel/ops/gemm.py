import torch
from typing import Optional
from . import _C


def cutlass_scaled_mm_bias_ls(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: Optional[torch.Tensor],
    ls: Optional[torch.Tensor],
) -> None:
    """Apply scaled mm on the given input, with optional bias and ls weight"""
    return _C.cutlass_scaled_mm(c, a, b, a_scales, b_scales, bias, ls)
