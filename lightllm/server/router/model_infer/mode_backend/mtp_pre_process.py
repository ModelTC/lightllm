import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput, ModelOutput

IS_NONE = -1


def prepare_mtp_prefill_inputs(
    req_objs: List[InferReq], model_input: ModelInput, last_hidden_states, tgt_input_ids, last_input_ids_cpu=None
):
    input_ids = []
    for i, req in enumerate(req_objs):
        if last_input_ids_cpu is None:
            input_token_ids = req.get_input_token_ids()
        else:
            input_token_ids = last_input_ids_cpu[i]
        input_token_ids = np.roll(input_token_ids, -1)
        input_token_ids[-1] = tgt_input_ids[i]
        input_ids.append(input_token_ids[req.cur_kv_len :])
    input_ids_cpu = input_ids
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    model_input.input_ids = input_ids
    # mtp embedding
    model_input.hidden_states = last_hidden_states
    return model_input, input_ids_cpu


def prepare_mtp_chunked_prefill_inputs(
    req_objs: List[InferReq],
    model_input: ModelInput,
    last_hidden_states,
    tgt_input_ids,
    shift,
    prev_step_has_output,
    last_input_ids_cpu=None,
    pad_for_empty_batch=False,
):
    if len(req_objs) == 0:
        assert pad_for_empty_batch
        model_input.hidden_states = last_hidden_states
        return model_input, input_ids_cpu, prev_step_has_output
    
    input_ids = []
    for i, req in enumerate(req_objs):
        if last_input_ids_cpu is None or not prev_step_has_output[i]:
            input_token_ids, is_last_chunked = req.get_chunked_input_token_ids_shift(shift)
            if prev_step_has_output[i]:
                input_token_ids[-1] = tgt_input_ids[i]
            prev_step_has_output[i] = is_last_chunked
        else:
            input_token_ids = last_input_ids_cpu[i]
            input_token_ids = np.roll(input_token_ids, -1)
            input_token_ids[-1] = tgt_input_ids[i]
            prev_step_has_output[i] = True
        input_ids.append(input_token_ids[req.cur_kv_len :])
    input_ids_cpu = input_ids
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    model_input.input_ids = input_ids
    # mtp embedding
    model_input.hidden_states = last_hidden_states
    return model_input, input_ids_cpu, prev_step_has_output


def prepare_draft_main_model_decode_inputs(req_objs: List[InferReq], draft_token_id_map, pad_for_empty_batch: bool = False):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_seq_len = []
    for req in req_objs:
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        input_id = req.get_last_gen_token()
        seq_len = req.get_cur_total_len()
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)

        # mtp step
        for step in range(draft_token_id_map.shape[1]):
            run_reqs.append(req)
            nopad_b_req_idx.append(req.req_idx)
            seq_len = req.get_cur_total_len() + step + 1
            assert req.cur_kv_len == seq_len - step - 2
            nopad_b_seq_len.append(seq_len)
            input_ids.append(draft_token_id_map[req.req_idx][step])
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)

    padded_req_num = 0
    if pad_for_empty_batch and len(req_objs) == 0:
        padded_req_num = 1
        nopad_b_req_idx.append(g_infer_context.req_manager.HOLD_REQUEST_ID)
        seq_len = 2
        nopad_b_seq_len.append(seq_len)
        input_ids.append(1)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
            
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    num_tokens_to_alloc = input_ids.shape[0] - padded_req_num 
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(num_tokens_to_alloc)
    mem_indexes_cpu = g_infer_context.req_manager.mem_manager.alloc(num_tokens_to_alloc)
    g_infer_state_lock.release()
    
    # 如果有填充请求，需要将填充请求的 mem_indexes 设置为 HOLD_TOKEN_MEMINDEX
    if padded_req_num > 0:
        padding_mem_indexs = torch.full(
            (padded_req_num,),
            fill_value=g_infer_context.req_manager.mem_manager.HOLD_TOKEN_MEMINDEX,
            dtype=torch.int32,
            device="cpu",
        )
        mem_indexes_cpu = torch.cat((mem_indexes_cpu, padding_mem_indexs), dim=0)
        
    mem_indexes = mem_indexes_cpu.cuda()
    
    model_input = ModelInput(
        batch_size=len(run_reqs),
        total_token_num=nopad_total_token_num,
        max_len_in_batch=nopad_max_len_in_batch,
        input_ids=input_ids,
        mem_indexes=mem_indexes,
        b_req_idx=nopad_b_req_idx,
        b_seq_len=nopad_b_seq_len,
        is_prefill=False,
    )
    return model_input, run_reqs, mem_indexes_cpu
