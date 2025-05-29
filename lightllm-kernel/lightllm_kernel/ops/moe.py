import torch
from typing import Optional
from . import _C


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
