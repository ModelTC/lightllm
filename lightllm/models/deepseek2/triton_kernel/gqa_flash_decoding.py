import torch
import triton
from typing import List
from lightllm.utils.log_utils import init_logger
from .gqa_flash_decoding_config import MlaDecodeAttentionKernelConfig

logger = init_logger(__name__)


def gqa_token_decode_attention_flash_decoding(
    q,
    kv,
    infer_state,
    q_head_num,
    kv_lora_rank,
    q_rope_dim,
    softmax_scale,
    out=None,
    alloc_tensor_func=torch.empty,
    **run_config
):
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    calcu_shape = (batch_size, q_head_num, kv_lora_rank + q_rope_dim)
    kv_calcu_shape = (-1, 1, kv_lora_rank + q_rope_dim)

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

    STAGE2_BLOCK_SEQ = run_config["STAGE2_BLOCK_SEQ"]

    from .gqa_flash_decoding_stage1 import flash_decode_stage1
    from .gqa_flash_decoding_stage2 import flash_decode_stage2
    from .gqa_flash_decoding_stage3 import flash_decode_stage3

    o_tensor = alloc_tensor_func((q.shape[0], q.shape[1], kv_lora_rank), q.dtype, q.device) if out is None else out

    mid_out_logics = alloc_tensor_func(
        [batch_size, q_head_num, max_len_in_batch],
        dtype=torch.float32,
        device="cuda",
    )
    mid_o_logexpsum = alloc_tensor_func(
        [batch_size, q_head_num, triton.cdiv(max_len_in_batch, STAGE2_BLOCK_SEQ)], dtype=torch.float32, device="cuda"
    )
    mid_o = alloc_tensor_func(
        [batch_size, q_head_num, triton.cdiv(max_len_in_batch, STAGE2_BLOCK_SEQ), kv_lora_rank],
        dtype=torch.float32,
        device="cuda",
    )

    flash_decode_stage1(
        q.view(calcu_shape),
        kv.view(kv_calcu_shape),
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_out_logics,
        softmax_scale,
        **run_config
    )

    flash_decode_stage2(
        mid_out_logics,
        kv.view(kv_calcu_shape),
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        infer_state.max_len_in_batch,
        mid_o,
        mid_o_logexpsum,
        q_rope_dim,
        **run_config
    )

    flash_decode_stage3(mid_o, mid_o_logexpsum, infer_state.b_seq_len, o_tensor, **run_config)
    return o_tensor
