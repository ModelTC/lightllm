import re
import torch
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferBatch, group_mapping, requests_mapping
from lightllm.common.basemodel.triton_kernel.apply_penalty import apply_penalty
from lightllm.server.router.model_infer.mode_backend.continues_batch.post_process import _get_post_sample_tensors, _top_p_top_k, random_sample


def sample(logits, reqs, is_prefill, vocab_size, req_manager, eos_id: List[int] = [2]):
    logits = logits.contiguous()
    (
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        exponential_decay_length_penalties,
        temperatures,
        top_ps,
        top_ks,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
        length_penalty_idx,
        mask_eos_reqs,
    ) = _get_post_sample_tensors(reqs)

    apply_penalty(
        logits,
        presence_penalties,
        frequency_penalties,
        repetition_penalties,
        p_token_ids,
        p_token_counts,
        p_cumsum_seq_len,
        p_max_len_in_batch,
    )
    logits[:, eos_id] = logits[:, eos_id] + torch.abs(logits[:, eos_id]) * (
        torch.pow(exponential_decay_length_penalties, length_penalty_idx).view((-1, 1)) - 1
    )
    logits[mask_eos_reqs, eos_id] = -1000000.0
    logits.div_(temperatures.view((-1, 1)))
    probs = torch.softmax(logits, dim=-1)

    batch_next_token_ids = []
    batch_next_token_logprobs = []
    start = 0
    for i in range(len(reqs)):
        req = reqs[i]
        best_of = req.sampling_param.best_of
        end = start + 1 if is_prefill else start + best_of
        if best_of > 1:
            next_token_id, next_token_logprob = beam_sample(probs[start:end], group_mapping[req.group_req_id], is_prefill, eos_id, vocab_size, req_manager)
        else:
            probs_sort, probs_idx = _top_p_top_k(probs[start:end], top_ps[start:end], top_ks[start:end])
            next_token_id, next_token_prob = random_sample(probs_sort, probs_idx)
            next_token_id = next_token_id.view(-1).detach().cpu().numpy()
            next_token_logprob = torch.log(next_token_prob).view(-1).detach().cpu().numpy()


        batch_next_token_ids.append(next_token_id)
        batch_next_token_logprobs.append(next_token_logprob)
        start = end
    return batch_next_token_ids, batch_next_token_logprobs

def beam_sample(probs, req_group, is_prefill, eos_id, vocab_size, req_manager):
    best_of = req_group.best_of
    next_token_id = []
    next_token_logprob = []
    valid_beams = 0
    logprobs = torch.log(probs)
    # print(req_group.cum_logprob)
    if not is_prefill:
        logprobs += req_group.cum_logprob.unsqueeze(1)
    # probs = probs.view(-1)
    next_logprobs, next_inds = torch.topk(logprobs.view(-1), 2 * num_beams, dim=0, largest=True, sorted=True)
    next_logprobs = next_logprobs.detach().cpu().numpy()
    beam_id = (next_inds // vocab_size).detach().cpu().numpy()
    next_tokens = (next_inds % vocab_size).detach().cpu().numpy()
    best_score = -float("inf")
    for i in range(2 * num_beams):
        req_obj = requests_mapping[req_group.req_group[valid_beams]]
        req_obj.cur_kv_len = len(req_obj.input_token_ids)
        req_obj.input_token_ids.append(next_tokens[i])
        req_obj.out_token_id_count[next_tokens[i]] += 1
        req_obj.update_finish_status(self.eos_id)

        if req_obj.finish_status.is_finished():
            output_ids = req_obj.input_token_ids[req_obj.prompt_len:]
            req_group.add_res(output_ids, next_logprobs[i], req_obj.finish_status.value)
            continue
        req_group.prev_beamid[valid_beams] = beam_id[i]
        req_group.cum_logprob[valid_beams] = float(next_logprobs[i])
        next_token_id.append(next_tokens[i])
        next_token_logprob.append(next_logprobs[i])
        best_score = max(next_logprobs[i] / req.get_output_len(), best_score)
        valid_beams += 1
        if valid_beams == num_beams:
            break
    req_manager.beam_copy(req_group.req_group, is_prefill)
    req_group.beam_copy()
    req_group.update_finish_status(best_score)
    return next_token_id, next_token_logprob