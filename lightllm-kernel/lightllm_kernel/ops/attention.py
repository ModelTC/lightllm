import torch
from typing import Optional, Tuple
from . import _C


def group8_int8kv_flashdecoding_stage1(
    seq_block_size: int,
    mid_o_emb: torch.Tensor,
    mid_o_logexpsum: torch.Tensor,
    att_scale: float,
    q: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    v: torch.Tensor,
    v_s: torch.Tensor,
    req_to_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_in_batch: int,
) -> None:

    return _C.group8_int8kv_flashdecoding_stage1(
        seq_block_size,
        mid_o_emb,
        mid_o_logexpsum,
        att_scale,
        q,
        k,
        k_s,
        v,
        v_s,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        max_len_in_batch,
    )


def group_int8kv_decode_attention(
    o: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    k_s: torch.Tensor,
    v: torch.Tensor,
    v_s: torch.Tensor,
    req_to_tokens: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_in_batch: int,
) -> None:

    return _C.group_int8kv_decode_attention(
        o,
        q,
        k,
        k_s,
        v,
        v_s,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        max_len_in_batch,
    )
