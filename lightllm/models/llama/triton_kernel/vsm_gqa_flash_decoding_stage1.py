import torch 
import triton 
import triton.language as tl
from lightllm.utils.device_utils import calcu_kernel_best_vsm_count
from .vsm_gqa_flash_decoding_config import VSMGQADecodeAttentionKernelConfig

@triton.jit
def _kernel_vsm_gqa_flash_decoding_stage1(
    Q,
    K,
    V,
    sm_scale,
    req_to_tokens,
    b_req_idx,
    b_seq_len,
    mid_o,
    mid_o_logexpsum,
    mid_o_batch_start_index,
    mid_o_chunk_num,
    total_token_in_the_batch_ptr,
    chunk_size_ptr,
    num_vsm,
    stride_q_bs,
    stride_q_h,
    stride_q_d,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_v_bs,
    stride_v_h,
    stride_v_d,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_mid_o_h,
    stride_mid_o_s,
    stride_mid_o_d,
    stride_mid_o_logexpsum_h,
    stride_mid_o_logexpsum_s,
    gqa_group_size,
    batch_size,
    GROUP_Q_HEAD_NUM: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    KV_HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    r'''
        q: [batch, q_head_num, q_head_dim]
        k: [batch, kv_head_dim, seq_len]

        grid for a kv-chunk 
    '''
    vsm_id = tl.program_id(0).to(tl.int64)
    out_batch_start_idx = tl.cast(0, tl.int64)
    total_token_in_the_batch = tl.load(total_token_in_the_batch_ptr, eviction_policy="evict_last")
    
    chunk_size = tl.cast(total_token_in_the_batch // num_vsm, dtype=tl.int32) + 1
    chunk_size = tl.cdiv(chunk_size, BLOCK_N) * BLOCK_N
    
    if vsm_id == 0:
        tl.store(chunk_size_ptr, chunk_size)

    group_q_head_offset = tl.arange(0, GROUP_Q_HEAD_NUM)
    off_q_dim = tl.arange(0, Q_HEAD_DIM)
    off_kv_dim = tl.arange(0, KV_HEAD_DIM)


    for cur_batch in range(batch_size):
        cur_batch_seq_len = tl.load(b_seq_len + cur_batch)
        cur_batch_req_idx = tl.load(b_req_idx + cur_batch)
        cur_block_num = tl.cdiv(cur_batch_seq_len, BLOCK_N) * gqa_group_size

        if vsm_id == 0:
            tl.store(mid_o_batch_start_index + cur_batch, out_batch_start_idx)           
            tl.store(mid_o_chunk_num + cur_batch, cur_block_num)
        cur_kv_chunk_idx = vsm_id
        while cur_kv_chunk_idx < cur_block_num:
            cur_kv_head = cur_kv_chunk_idx % gqa_group_size
            cur_end_chunk_idx = cur_kv_chunk_idx // gqa_group_size
            cur_kv_chunk_idx += num_vsm

            cur_q_head = cur_kv_head * gqa_group_size + group_q_head_offset

            cur_batch_start_index = cur_end_chunk_idx * chunk_size
            cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + chunk_size)
            cur_kv_chunk_num = tl.cdiv(cur_batch_end_index - cur_batch_start_index, BLOCK_N)

            off_q = cur_batch * stride_q_bs + cur_q_head[:, None] * stride_q_h + off_q_dim[None, :] # shape [GROUP_Q_HEAD_NUM, Q_HEAD_DIM]
            q = tl.load(Q + off_q, mask=(cur_q_head[:, None] < (cur_kv_head + 1) * gqa_group_size), other=0.0)

            
            sum_exp = tl.zeros([GROUP_Q_HEAD_NUM], dtype=tl.float32)
            max_exp = tl.zeros([GROUP_Q_HEAD_NUM], dtype=tl.float32) - float('inf')
            acc = tl.zeros([GROUP_Q_HEAD_NUM, Q_HEAD_DIM], dtype=tl.float32)

            for cur_kv_chunk in tl.range(0, cur_kv_chunk_num, 1, num_stages=NUM_STAGES):
                off_token = cur_kv_chunk * BLOCK_N + tl.arange(0, BLOCK_N)
                off_req = cur_batch_req_idx * stride_req_to_tokens_b + off_token
                mask_kv = (off_token < cur_batch_end_index)
                kv_loc = tl.load(
                    req_to_tokens + off_req,
                    mask=mask_kv,
                    other=0
                )

                off_kv = kv_loc[None, :] * stride_k_bs + cur_kv_head * stride_k_h + off_kv_dim[:, None] # shape: (d, chunk)
                mask_kv = (off_token)
                k = tl.load(K + off_kv, 
                        mask=mask_kv[None, :],
                        other=0.0)
                att_value = tl.dot(q, k)
                att_value *= sm_scale
                att_value = tl.where(mask_kv[None, :], att_value, float('-inf'))
                off_kv_trans = kv_loc[:, None] * stride_k_bs + cur_kv_head * stride_k_h + off_kv_dim[None, :] # shape: (d, chunk)
                v = tl.load(V + off_kv_trans, 
                            mask=mask_kv[:, None],
                            other=0.0)
                cur_max = tl.max(att_value, axis=1)
                new_max = tl.maximum(cur_max, max_exp)

                exp_norm = tl.exp(att_value - new_max[:, None])
                exp_scale = tl.exp(max_exp - new_max)
                acc *= exp_scale[:, None]
                acc += tl.dot(exp_norm.to(v.dtype), v)

                sum_exp = sum_exp * exp_scale + tl.sum(exp_norm, axis=1)
                max_exp = new_max
            
            head_mask = cur_q_head[:, None] < (cur_kv_head + 1) * gqa_group_size
            off_mid_o = cur_q_head * stride_mid_o_h + (out_batch_start_idx + cur_end_chunk_idx) * stride_mid_o_s + off_q_dim
            off_mid_log_expsum = cur_q_head * stride_mid_o_logexpsum_h + (out_batch_start_idx + cur_end_chunk_idx) * stride_mid_o_logexpsum_s
            tl.store(mid_o + off_mid_o, acc / sum_exp[:, None], mask=head_mask)
            tl.store(mid_o_logexpsum + off_mid_log_expsum, max_exp + tl.log(sum_exp), mask=head_mask)

        out_batch_start_idx += cur_block_num // gqa_group_size

def vsm_gqa_flash_decoding_stage1(
    q,
    k,
    v,
    req_to_token_indexs,
    b_req_idx,
    b_seq_len,
    total_token_in_the_batch_addr, # shape [1]
    mid_o,
    mid_o_logexpsum,
    mid_o_batch_start_index,
    mid_o_chunk_num,
    num_vsm,
    get_sm_count,
    **run_config: VSMGQADecodeAttentionKernelConfig
):
    BLOCK_N = run_config["BLOCK_N"]
    BLOCK_Q_HEAD = run_config["BLOCK_Q_HEAD"]
    num_warps = run_config["stage1_num_warps"]
    num_stages = run_config["stage1_num_stages"]

    sm_scale = 1. / (k.shape[-1] ** 0.5)
    gqa_group_size = q.shape[1] // k.shape[1]
    batch_size = b_req_idx.shape[0]
    GROUP_Q_HEAD_NUM =max(16, triton.next_power_of_2(gqa_group_size))
    Q_HEAD_DIM = q.shape[-1]
    KV_HEAD_NUM = k.shape[1]

    kernel = _fwd_kernel_vsm_gqa_flash_decoding_stage1.warmup(
        q.
        k,
        v,
        sm_scale,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        mid_o,
        mid_o_logexpsum,
        mid_o_batch_start_index,
        mid_o_chunk_num,
        total_token_in_the_batch_addr,
        num_vsm,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *req_to_token_indexs.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        gqa_group_size=gqa_group_size,
        batch_size=batch_size,
        GROUP_Q_HEAD_NUM=GROUP_Q_HEAD_NUM,
        Q_HEAD_DIM=Q_HEAD_DIM,
        KV_HEAD_NUM=KV_HEAD_NUM,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps=1,
        grid=(1,)
    )
    kernel._init_handles()
    num_vsm = calcu_kernel_best_vsm_count(kernel, num_warps=num_warps)
    if get_sm_count:
        return num_vsm
    grid = (num_vsm, )

    _fwd_kernel_vsm_gqa_flash_decoding_stage1[grid](
        q.
        k,
        v,
        sm_scale,
        req_to_token_indexs,
        b_req_idx,
        b_seq_len,
        mid_o,
        mid_o_logexpsum,
        mid_o_batch_start_index,
        mid_o_chunk_num,
        total_token_in_the_batch_addr,
        num_vsm,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *req_to_token_indexs.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        gqa_group_size=gqa_group_size,
        batch_size=batch_size,
        GROUP_Q_HEAD_NUM=GROUP_Q_HEAD_NUM,
        Q_HEAD_DIM=Q_HEAD_DIM,
        KV_HEAD_NUM=KV_HEAD_NUM,
        BLOCK_N=BLOCK_N,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    
