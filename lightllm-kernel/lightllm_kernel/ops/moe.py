import torch
from typing import Optional
from . import _C


def all_gather(
    _fa: int, inp: torch.Tensor, out: torch.Tensor, _reg_buffer: int, reg_buffer_sz_bytes: int
) -> torch.Tensor:
    """Apply rmsnorm on given X, with weight W and eps"""
    return _C.all_gather(_fa, inp, out, _reg_buffer, reg_buffer_sz_bytes)


def grouped_topk(
    topk_weights: torch.Tensor,
    correction_bias: torch.Tensor,
    topk_indices: torch.Tensor,
    group_indices: torch.Tensor,
    gating_output: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    scoring_func: str,
    group_scores: torch.Tensor,
) -> torch.Tensor:
    """Apply rmsnorm on given X, with weight W and eps"""
    return _C.grouped_topk(
        topk_weights,
        correction_bias,
        topk_indices,
        group_indices,
        gating_output,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        scoring_func,
        group_scores,
    )
