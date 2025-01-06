import torch
from .vsm_gqa_flash_decoding_config import VSMGQADecodeAttentionKernelConfig
from .vsm_gqa_flash_decoding_stage1 import vsm_gqa_flash_decoding_stage1
from .vsm_gqa_flash_decoding_stage2 import vsm_gqa_flash_decoding_stage2

def vsm_gqa_flash_decoding(
    q, 
    infer_state,
    k, 
    v,
    q_head_dim,
    q_head_num,
    kv_head_dim,
    kv_head_num,
    out=None,
    alloc_tensor_func=torch.empty,
    **run_config: VSMGQADecodeAttentionKernelConfig
):
    batch_size = infer_state.batch_size
    max_len_in_batch = infer_state.max_len_in_batch
    total_token_in_the_batch = infer_state.total_token_num

    q_shape = (batch_size, q_head_num, q_head_dim)

    if not run_config:
        if torch.cuda.is_current_stream_capturing():
            avg_seq_len_in_batch = max_len_in_batch
        else:
            avg_seq_len_in_batch = total_token_in_the_batch // batch_size
        run_config = VSMGQADecodeAttentionKernelConfig.try_to_get_best_config(
            batch_size=batch_size,
            avg_seq_len_in_batch=avg_seq_len_in_batch,
            q_head_num=q_head_num,
            q_head_dim=q_head_dim,
            kv_head_num=kv_head_num,
            kv_head_dim=kv_head_dim,
            out_dtype=torch.bfloat16,
        )

    o_tensor = alloc_tensor_func(q.shape, dtype=q.dtype, device=q.device) if out is None else out

    mid_o_block_seq = torch.empty([1], dtype=torch.int64, device="cuda")
    mid_o_batch_start_index = alloc_tensor_func(
        [
            batch_size,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    chunk_size = torch.empty([1], dtype=torch.int32, device="cuda")
    # virtual calculate

    mid_o = torch.empty([q_head_num, 0, kv_head_dim], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, 0], dtype=torch.float32, device="cuda")
    vsm_count = vsm_gqa_flash_decoding_stage1(
        q.view(q_shape),
        k,
        v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        total_token_in_the_batch,
        mid_o,
        mid_o_logexpsum,
        mid_o_block_seq,
        mid_o_batch_start_index,
        chunk_size,
        num_vsm=1,
        get_sm_count=True,
        **run_config
    )

    # real calculate
    mid_o = torch.empty([q_head_num, vsm_count * 4 + batch_size, kv_head_dim], dtype=torch.float32, device="cuda")
    mid_o_logexpsum = torch.empty([q_head_num, vsm_count * 4 + batch_size], dtype=torch.float32, device="cuda")
    vsm_gqa_flash_decoding_stage1(
        q.view(q_shape),
        k,
        v,
        infer_state.req_manager.req_to_token_indexs,
        infer_state.b_req_idx,
        infer_state.b_seq_len,
        total_token_in_the_batch,
        mid_o,
        mid_o_logexpsum,
        mid_o_block_seq,
        mid_o_batch_start_index,
        chunk_size,
        num_vsm=vsm_count,
        get_sm_count=False,
        **run_config
    )

    vsm_gqa_flash_decoding_stage2(
        mid_o,
        mid_o_logexpsum,
        mid_o_block_seq,
        mid_o_batch_start_index,
        infer_state.b_seq_len,
        chunk_size,
        o_tensor.view(q_shape),
        **run_config
    )
    return o_tensor


    