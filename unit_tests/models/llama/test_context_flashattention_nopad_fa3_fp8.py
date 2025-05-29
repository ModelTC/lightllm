import torch
import time
import pytest
import numpy as np
import torch.nn.functional as F
from lightllm.utils.log_utils import init_logger
from lightllm.models.llama.triton_kernel.context_flashattention_nopad import (
    context_attention_fwd,
)
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.req_manager import ReqManager
from lightllm.utils.sgl_utils import flash_attn_with_kvcache

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def q_quantize_per_head_fp8(q: torch.Tensor, seq_lens):
    min_fp8 = torch.finfo(torch.float8_e4m3fn).min
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    _, H, _, _ = q.shape
    splits = torch.split(q, seq_lens.tolist(), dim=0)
    max_list = [sp.abs().amax(dim=(0, 2, 3)) for sp in splits]
    max_per_bh = torch.stack(max_list, dim=0)  # [B, H]
    scales = torch.where(max_per_bh > 0, max_per_bh / max_fp8, torch.ones_like(max_per_bh)).to(torch.float32)
    q_list = []
    for b, sp in enumerate(splits):
        scale_b = scales[b].view(1, H, 1, 1)
        q_sp = (sp / scale_b).round().clamp(min_fp8, max_fp8).to(torch.float8_e4m3fn)
        q_list.append(q_sp)
    q_q = torch.cat(q_list, dim=0)  # [T, R, H, D]
    return q_q, scales


def kv_quantize_per_head_fp8(kv_buffer: torch.Tensor, seq_lens):
    device = kv_buffer.device
    B = seq_lens.size(0)
    min_fp8 = torch.finfo(torch.float8_e4m3fn).min
    max_fp8 = torch.finfo(torch.float8_e4m3fn).max
    _, S_max, H, D = kv_buffer.shape
    seq_range = torch.arange(S_max, device=device)[None, :]
    valid_mask = (seq_range < seq_lens[:, None]).view(B, S_max, 1, 1)
    masked = kv_buffer * valid_mask
    max_per_bh = masked.abs().amax(dim=(1, 3))  # [B, H]
    scales = torch.where(max_per_bh > 0, max_per_bh / max_fp8, torch.ones_like(max_per_bh)).to(torch.float32)
    scales_exp = scales.view(B, 1, H, 1)
    q = (kv_buffer / scales_exp).round().clamp(min_fp8, max_fp8).to(torch.float8_e4m3fn)
    return q, scales


@pytest.mark.parametrize(
    "batch, seqlen, q_heads, kv_heads, head_dim",
    [
        (a, b, c, d, e)
        for a in [1, 16, 32, 128, 512]
        for b in [16, 32, 512, 1024]
        for c in [28]
        for d in [4]
        for e in [128]
    ],
)
def test_context_attention_fwd_fa3_fp8(batch, seqlen, q_heads, kv_heads, head_dim):
    Z, N_CTX, Q_HEADS, KV_HEADS, HEAD_DIM = batch, seqlen, q_heads, kv_heads, head_dim
    dtype = torch.bfloat16
    kv = torch.randn((Z * N_CTX, 2 * KV_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * (N_CTX // 2)
    rand_num = torch.randint_like(b_seq_len, high=(N_CTX // 2), dtype=torch.int32, device="cuda")
    b_seq_len += rand_num
    b_ready_cache_len = torch.zeros_like(b_seq_len, dtype=torch.int32, device="cuda")
    if N_CTX > 1:
        b_ready_cache_len = torch.randint_like(b_seq_len, high=(N_CTX - 1) // 2, dtype=torch.int32, device="cuda")
    b_req_idx = torch.randperm(Z, dtype=torch.int32).cuda()
    q_lens = b_seq_len - b_ready_cache_len
    q_start_loc = q_lens.cumsum(0) - q_lens

    q = torch.randn((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o = torch.zeros((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")
    o1 = torch.zeros((q_lens.sum(), Q_HEADS, HEAD_DIM), dtype=dtype, device="cuda")

    infer_state = LlamaInferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
    infer_state.req_manager = ReqManager(Z, N_CTX, None)
    infer_state.req_manager.req_to_token_indexs = req_to_token_indexs
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len
    infer_state.b_ready_cache_len = b_ready_cache_len
    infer_state.b_start_loc = q_start_loc

    context_attention_fwd(
        q,
        kv[:, :KV_HEADS, :],
        kv[:, KV_HEADS:, :],
        o,
        infer_state.b_req_idx,
        infer_state.b_start_loc,
        infer_state.b_seq_len,
        infer_state.b_ready_cache_len,
        infer_state.max_len_in_batch,
        infer_state.req_manager.req_to_token_indexs,
    )

    batch_size = Z
    head_dim = HEAD_DIM
    q_heads = Q_HEADS
    kv_heads = KV_HEADS
    page_table = torch.empty((batch_size, N_CTX), dtype=torch.int32).to(0)
    page_table.copy_(req_to_token_indexs[b_req_idx, :N_CTX])

    q_starts = torch.zeros((Z + 1,)).int().cuda()
    q_starts[1:] = torch.cumsum(b_seq_len - b_ready_cache_len, dim=0)
    kv_starts = torch.zeros_like(q_starts)
    kv_starts[1:] = torch.cumsum(b_seq_len, dim=0)

    k_cache = kv[:, :KV_HEADS, :].contiguous()
    v_cache = kv[:, KV_HEADS:, :].contiguous()
    # o1 = flash_attn_with_kvcache(
    #     q=q,
    #     k_cache=k_cache[page_table].view(-1, N_CTX, kv_heads, head_dim),
    #     v_cache=v_cache[page_table].view(-1, N_CTX, kv_heads, head_dim),
    #     # page_table=page_table,
    #     cache_seqlens=infer_state.b_seq_len,
    #     cu_seqlens_q=q_starts,
    #     cu_seqlens_k_new=kv_starts,
    #     max_seqlen_q=N_CTX,
    #     causal=True,
    #     window_size=(-1, -1),
    #     softcap=0.0,
    #     return_softmax_lse=False,
    # )

    q, q_scale = q_quantize_per_head_fp8(q.view(q.shape[0], kv_heads, -1, head_dim), q_lens)
    k, k_scale = kv_quantize_per_head_fp8(k_cache[page_table], b_seq_len)
    v, v_scale = kv_quantize_per_head_fp8(v_cache[page_table], b_seq_len)
    o1 = flash_attn_with_kvcache(
        q=q.view(-1, q_heads, head_dim),
        k_cache=k.view(-1, N_CTX, kv_heads, head_dim),
        v_cache=v.view(-1, N_CTX, kv_heads, head_dim),
        # page_table=page_table,
        cache_seqlens=infer_state.b_seq_len,
        cu_seqlens_q=q_starts,
        cu_seqlens_k_new=kv_starts,
        max_seqlen_q=N_CTX,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        q_descale=q_scale.view(batch_size, kv_heads),
        k_descale=k_scale.view(batch_size, kv_heads),
        v_descale=v_scale.view(batch_size, kv_heads),
        return_softmax_lse=False,
    )

    # assert torch.allclose(o, o1, atol=1e-2, rtol=0)
    cos_sim1 = F.cosine_similarity(o, o1).mean()
    print(cos_sim1)
    assert cos_sim1.item() == 1


if __name__ == "__main__":
    test_context_attention_fwd_fa3_fp8(16, 1024, 28, 4, 128)
