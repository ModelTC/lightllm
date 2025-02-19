import os
import torch
import torch.multiprocessing as mp
import triton
import triton.language as tl
from typing import List
from lightllm.utils.log_utils import init_logger
from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig
from lightllm.utils.device_utils import get_device_sm_count

logger = init_logger(__name__)


def gqa_token_decode_attention_flash_decoding(
    q_nope,
    q_rope,
    kv_nope,
    kv_rope,
    infer_state,
    q_head_num,
    kv_lora_rank,
    q_rope_dim,
    qk_nope_head_dim,
    softmax_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    **run_config
):
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape1 = (batch_size, q_head_num, kv_lora_rank)
    calcu_shape2 = (batch_size, q_head_num, q_rope_dim)

    if not run_config:
        if torch.cuda.is_current_stream_capturing():
            avg_seq_len_in_batch = max_len_in_batch
        else:
            avg_seq_len_in_batch = infer_state.total_token_num // batch_size

        run_config = MlaDecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=avg_seq_len_in_batch,
            q_head_num=q_head_num,
            q_head_dim=kv_lora_rank,
            q_rope_dim=q_rope_dim,
            out_dtype=torch.bfloat16,
        )

    BLOCK_N = run_config["BLOCK_N"]

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2

    o_tensor = alloc_tensor_func(q_nope.shape, q_nope.dtype, q_nope.device) if out is None else out

    fake_decode_att_block_seq = torch.empty([0], dtype=torch.int64, device="cuda")
    mid_o = torch.empty([q_head_num, 0, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda")

    vsm_count = flash_decode_stage1(
        fake_decode_att_block_seq,
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        get_sm_count=True,
        **run_config
    )

    if not hasattr(infer_state, "decode_att_block_seq"):
        assert batch_size <= 2048
        decode_att_block_seq = torch.empty(
            [
                1,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        mid_o_batch_start_index = torch.empty(
            [
                batch_size,
            ],
            dtype=torch.int64,
            device="cuda",
        )
        _fwd_kernel_calcu_index_and_block_seq[(1,)](
            infer_state.b_seq_len,
            decode_att_block_seq,
            mid_o_batch_start_index,
            vsm_count,
            batch_size,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        infer_state.decode_att_block_seq = decode_att_block_seq
        infer_state.mid_o_batch_start_index = mid_o_batch_start_index

    mid_o = torch.empty([q_head_num, vsm_count * 4 + batch_size, kv_lora_rank], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, vsm_count * 4 + batch_size], dtype=torch.float32, device="cuda")

    flash_decode_stage1(
        infer_state.decode_att_block_seq,
        q_nope.view(calcu_shape1),
        q_rope.view(calcu_shape2),
        kv_nope,
        kv_rope,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        mid_o,
        mid_o_logexpsum,
        softmax_scale,
        get_sm_count=False,
        **run_config
    )

    flash_decode_stage2(
        infer_state.decode_att_block_seq,
        infer_state.mid_o_batch_start_index,
        mid_o,
        mid_o_logexpsum,
        infer_state.b_seq_len,
        o_tensor.view(calcu_shape1),
        **run_config
    )
    return o_tensor


@triton.jit
def _fwd_kernel_calcu_index_and_block_seq(
    b_seq_len_ptr,
    mid_o_decode_att_block_seq_ptr,
    mid_o_batch_start_index_ptr,
    num_sm,
    batch_size,
    BLOCK_N: tl.constexpr,
):
    b_seq_len = tl.load(b_seq_len_ptr + tl.arange(0, 2048), mask=tl.arange(0, 2048) < batch_size, other=0)
    total_token_num = tl.sum(b_seq_len)

    block_seq = tl.cast(total_token_num / (num_sm * 4), dtype=tl.int32) + 1
    block_seq = tl.cdiv(block_seq, BLOCK_N) * BLOCK_N

    block_seq_len = tl.cdiv(b_seq_len, block_seq)
    cumsum_seq_len = tl.cumsum(block_seq_len)
    batch_start_index = cumsum_seq_len - block_seq_len
    tl.store(mid_o_batch_start_index_ptr + tl.arange(0, 2048), batch_start_index, mask=tl.arange(0, 2048) < batch_size)
    tl.store(mid_o_decode_att_block_seq_ptr, block_seq)
    return


if __name__ == "__main__":
    import torch
    import numpy as np
    import torch.nn.functional as F
    from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
    from lightllm.common.req_manager import ReqManager
    import flashinfer
    import lightllm_ppl_mla

    Z, N_CTX, H, D_HEAD, ROPE_HEAD = 10, 1024, 16, 512, 64
    dtype = torch.bfloat16
    sm_scale = 1.0 / ((D_HEAD + ROPE_HEAD) ** 0.5)
    q_nope = torch.randn((Z, H, D_HEAD), dtype=dtype, device="cuda")
    q_rope = torch.randn((Z, H, ROPE_HEAD), dtype=dtype, device="cuda")

    kv = torch.randn((Z * N_CTX, 1, D_HEAD + ROPE_HEAD), dtype=dtype, device="cuda")
    kv_scale = torch.randn((Z * N_CTX, 1, 1), dtype=dtype, device="cuda")
    kv_fp8 = kv.to(torch.float8_e4m3fn)

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * N_CTX
    b_start_loc = torch.arange(Z).cuda().int() * N_CTX
    b_start_loc[0] = 0
    b_req_idx = torch.arange(Z).cuda().int()
    kv_starts = torch.cat([b_start_loc, b_start_loc[-1:] + b_seq_len[-1:]], dim=0)

    o = torch.zeros((Z, H, D_HEAD), dtype=dtype, device="cuda")
    o1 = torch.zeros((Z, H, D_HEAD), dtype=dtype, device="cuda")
    o2 = torch.zeros((Z, H, D_HEAD), dtype=dtype, device="cuda")

    infer_state = Deepseek2InferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
    infer_state.req_manager = ReqManager(Z, N_CTX, None)
    infer_state.req_manager.req_to_token_indexs = req_to_token_indexs
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len
    infer_state.kv_starts = kv_starts

    kv_nope = kv[:, :, :D_HEAD]
    kv_rope = kv[:, :, D_HEAD:]
    fn1 = lambda: gqa_token_decode_attention_flash_decoding(
        q_nope,
        q_rope,
        kv_nope,
        kv_rope,
        infer_state,
        H,
        D_HEAD,
        ROPE_HEAD,
        D_HEAD,
        sm_scale,
        o,
    )

    q = torch.cat([q_nope, q_rope], dim=-1)
    fn2 = lambda: lightllm_ppl_mla.decode_mla(
        o1,
        q,
        kv,
        req_to_token_indexs,
        kv_starts,
        b_req_idx,
        sm_scale,
        D_HEAD + ROPE_HEAD,
        D_HEAD,
    )

    batch_size = Z
    head_dim_ckv = D_HEAD
    head_dim_kpe = ROPE_HEAD
    num_heads = H
    page_size = 1
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    q_indptr = torch.arange(batch_size + 1).to(0).int()
    kv_indptr = infer_state.kv_starts
    kv_indices = torch.arange(Z * N_CTX).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        kv_indices[start : start + sl] = req_to_token_indexs[b][:sl]
    kv_lens = b_seq_len
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend="fa2",
        use_cuda_graph=True,
        qo_indptr=q_indptr,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_len_arr=kv_lens,
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        kv.dtype,
    )
    fn3 = lambda: wrapper.run(q_nope, q_rope, kv_nope, kv_rope, out=o2, return_lse=False)

    ms1 = triton.testing.do_bench_cudagraph(fn1)
    ms2 = triton.testing.do_bench_cudagraph(fn2)
    ms3 = triton.testing.do_bench_cudagraph(fn3)
    print(ms1)
    print(ms2)
    print(ms3)

    cos_sim1 = F.cosine_similarity(o, o1).mean()
    cos_sim2 = F.cosine_similarity(o, o2).mean()
    print(cos_sim1, cos_sim2)
