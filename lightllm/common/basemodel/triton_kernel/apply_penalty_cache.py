import torch

import triton
import triton.language as tl
import torch.nn.functional as F
import numpy as np


@triton.jit
def _fwd_kernel_apply_penalty_cache(
    Logits,
    req_idxs,
    presence_penalty,
    freqency_penalty,
    repetition_penalty,
    p_token_vocabs,
    stride_logit_b,
    stride_p_token_vocabs_b,
    BLOCK_P: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    block_idx = tl.program_id(1)
    token_idx = tl.load(req_idxs + cur_batch)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)

    batch_ids = BLOCK_P * block_idx + tl.arange(0, BLOCK_P)
    batch_ids_count = tl.load(
        p_token_vocabs + token_idx * stride_p_token_vocabs_b + batch_ids,
        mask=batch_ids < stride_p_token_vocabs_b,
        other=0,
    )
    row_start_ptr = Logits + cur_batch * stride_logit_b
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=batch_ids_count > 0, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * stride_logit_b + batch_ids
    tl.store(output_ptr, pre_logits, mask=batch_ids_count > 0)
    return


@triton.jit
def _eos_penalty(
    Logits,
    p_token_lens,
    exponential_decay_length_penalties,
    length_penalty_idx,
    eos_ids,
    mask_eos_reqs,
    stride_logit_b,
    EOS_ID_NUM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    exponential_decay_length_penalty = tl.load(exponential_decay_length_penalties + cur_batch)
    token_lens = tl.load(p_token_lens + cur_batch)
    length_penalty = tl.maximum(token_lens - tl.load(length_penalty_idx + cur_batch), 0)
    penalty_scale = tl.exp2(tl.log2(exponential_decay_length_penalty) * length_penalty) - 1
    mask_eos = tl.load(mask_eos_reqs + cur_batch)
    for eos_index in range(EOS_ID_NUM):
        eos_id = tl.load(eos_ids + eos_index)
        cur_eos_logit_ptr = Logits + cur_batch * stride_logit_b + eos_id
        cur_eos_logit = tl.load(cur_eos_logit_ptr)
        cur_eos_logit = cur_eos_logit + tl.abs(cur_eos_logit) * penalty_scale
        cur_eos_logit = tl.where(token_lens < mask_eos, -10000000.0, cur_eos_logit)
        tl.store(cur_eos_logit_ptr, cur_eos_logit)
    return


@torch.no_grad()
def apply_penalty_cache(
    Logits,
    req_idxs,
    presence_penalty,
    freqency_penalty,
    repetition_penalty,
    p_token_vocabs,
    p_token_lens,
    exponential_decay_length_penalties,
    length_penalty_idx,
    eos_ids,
    mask_eos_reqs,
    is_eos_penalty=False,
):
    assert Logits.is_contiguous()
    BLOCK_P = 1024
    num_warps = 8
    vocab_size = p_token_vocabs.shape[1]
    _fwd_kernel_apply_penalty_cache[(Logits.shape[0], triton.cdiv(vocab_size, BLOCK_P))](
        Logits,
        req_idxs,
        presence_penalty,
        freqency_penalty,
        repetition_penalty,
        p_token_vocabs,
        Logits.stride(0),
        p_token_vocabs.stride(0),
        num_warps=num_warps,
        BLOCK_P=BLOCK_P,
    )
    if is_eos_penalty:
        p_token_lens = p_token_vocabs[req_idxs].count_nonzero(dim=1) if p_token_lens is None else p_token_lens
        _eos_penalty[(Logits.shape[0],)](
            Logits,
            p_token_lens,
            exponential_decay_length_penalties,
            length_penalty_idx,
            eos_ids,
            mask_eos_reqs,
            Logits.stride(0),
            num_warps=num_warps,
            EOS_ID_NUM=eos_ids.shape[0],
        )
    return


if __name__ == "__main__":
    from .apply_penalty import apply_penalty

    bs = 200
    vocab_size = 150000
    p_tokens = 2000
    repseats = 1000
    req_idxs = torch.arange(bs).cuda()
    logits = torch.randn((bs, vocab_size), dtype=torch.float32).cuda()
    logits2 = logits.clone()

    presence_penalty = torch.randn((bs,), dtype=torch.float32).cuda() + 1e-5
    freqency_penalty = torch.randn((bs,), dtype=torch.float32).cuda()
    repetition_penalty = torch.randn((bs,), dtype=torch.float32).cuda()
    exponential_decay_length_penalties = torch.rand(bs).cuda()
    eos_ids = torch.tensor([999]).cuda()

    p_seq_len = torch.cat([torch.tensor([0]), torch.randint(1, p_tokens, (bs,))]).cuda()
    p_token_ids = torch.randint(0, vocab_size, (p_seq_len.sum(),)).cuda()
    i = 0
    for s_l in p_seq_len[1:]:
        p_token_ids[i : i + s_l] = torch.randperm(vocab_size)[:s_l]
        i += s_l
    p_token_counts = torch.randint(1, repseats, (p_seq_len.sum(),)).cuda()
    p_cumsum_seq_len = p_seq_len.cumsum(dim=0).cuda()
    p_token_vocabs = torch.zeros((bs, vocab_size), dtype=torch.int16).cuda()
    i = 0
    b = 0
    for token_id, count in zip(p_token_ids, p_token_counts):
        p_token_vocabs[b][token_id] = count
        i += 1
        if i == p_seq_len[b + 1]:
            b += 1
            i = 0

    p_token_lens = p_token_vocabs.sum(dim=1).cuda()
    length_penalty_idx = torch.randint(0, p_tokens, (bs,)).cuda()
    len_idx = torch.tensor([max(p_token_lens[i] - length_penalty_idx[i], 0) for i in range(bs)]).cuda()
    mask_eos_reqs = torch.randint(1, p_tokens, (bs,)).cuda()
    mask_bool = torch.tensor([p_token_lens[i] < mask_eos_reqs[i] for i in range(bs)]).cuda()

    fn1 = lambda: apply_penalty(
        logits,
        presence_penalty,
        freqency_penalty,
        repetition_penalty,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        exponential_decay_length_penalties,
        len_idx,
        eos_ids,
        mask_bool,
    )

    fn2 = lambda: apply_penalty_cache(
        logits2,
        req_idxs,
        presence_penalty,
        freqency_penalty,
        repetition_penalty,
        p_token_vocabs,
        p_token_lens,
        exponential_decay_length_penalties,
        length_penalty_idx,
        eos_ids,
        mask_eos_reqs,
    )
    fn1()
    fn2()
    cos = F.cosine_similarity(logits, logits2).mean()
    print("cos =", cos)
    assert torch.allclose(logits, logits2, atol=1e-2, rtol=0)

    ms1 = triton.testing.do_bench(fn1)
    ms2 = triton.testing.do_bench(fn2)
    print("ms1 =", ms1, "ms2 =", ms2)
