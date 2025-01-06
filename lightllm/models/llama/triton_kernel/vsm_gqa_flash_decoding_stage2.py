import triton 
import triton.language as tl
from .vsm_gqa_flash_decoding_config import VSMGQADecodeAttentionKernelConfig

@triton.jit
def _fwd_kernel_vsm_gqa_flash_decoding_stage2(
    mid_o, 
    mid_o_logexp_sum,
    mid_o_chunk_num,
    mid_o_batch_start,
    b_seq_len,
    chunk_size_ptr,
    out,
    stride_mid_o_h,
    stride_mid_o_s,
    stride_mid_o_d,
    stride_mid_o_logexpsum_h,
    stride_mid_o_logexpsum_s,
    stride_out_bs,
    stride_out_h,
    stride_out_d,
    Q_DIM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    cur_head = tl.program_id(0)
    cur_batch = tl.program_id(1)

    off_d = tl.arange(0, Q_DIM)
    cur_batch_seq_len = tl.load(b_seq_len + cur_batch)
    cur_chunk_start = tl.load(mid_o_batch_start + cur_batch)
    cur_chunk_num = tl.load(mid_o_chunk_num)
    chunk_size = tl.load(chunk_size_ptr)

    sum_exp = 0.
    max_term = -float('inf')
    acc = tl.zeros([Q_DIM], dtype=tl.float32)
    
    
    for cur_chunk in tl.range(0, cur_chunk_num, 1, num_stages=NUM_STAGES):
        off_m = cur_head * stride_mid_o_h + (cur_chunk_start + cur_chunk) * stride_mid_o_s + off_d
        off_ml = cur_head * stride_mid_o_logexpsum_h + (cur_chunk_start + cur_chunk)
        cur_mid = tl.load(mid_o + off_m)
        cur_ml = tl.load(mid_o_logexp_sum + off_ml)
        new_max = tl.maximum(cur_ml, max_term)

        old_scale = tl.exp(max_term - new_max)
        acc *= old_scale
        exp_ml = tl.exp(cur_ml - new_max)
        acc += exp_ml * cur_mid
        sum_exp = sum_exp * old_scale + exp_ml
        max_term = new_max
    tl.store(out + cur_batch + stride_out_bs + cur_head + stride_out_h + off_d, acc / sum_exp)
    

def vsm_gqa_flash_decoding_stage2(
    mid_o,
    mid_o_logexpsum,
    mid_o_chunk_num,
    mid_o_batch_start_index,
    b_seq_len,
    chunk_size,
    o,
    **run_config: VSMGQADecodeAttentionKernelConfig
):
    return
    num_warps = run_config['stage2_num_warps']
    num_stages = run_config['stage2_num_stages']

    Q_DIM = mid_o.shape[-1]
    batch_size, head_num = mid_o_batch_start_index.shape[0], mid_o.shape[0]
    grid = (head_num, batch_size)

    _fwd_kernel_vsm_gqa_flash_decoding_stage2[grid](
        mid_o,
        mid_o_logexpsum,
        mid_o_chunk_num,
        mid_o_batch_start_index,
        b_seq_len,
        chunk_size,
        o,
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        *o.stride(),
        Q_DIM=Q_DIM,
        NUM_STAGES=num_stages
    )