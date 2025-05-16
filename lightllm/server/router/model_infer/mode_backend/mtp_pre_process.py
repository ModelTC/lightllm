import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock

IS_NONE = -1


def prepare_mtp_prefill_inputs(req_objs: List[InferReq], tgt_input_ids, mem_manager):
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    for i, req in enumerate(req_objs):
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)

        input_token_ids = req.get_input_token_ids()

        input_token_ids[:-1] = input_token_ids[1:]
        input_token_ids[-1] = tgt_input_ids[i]

        seq_len = len(input_token_ids)
        input_token_len = seq_len

        input_id = input_token_ids

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.zeros(len(req_objs), dtype=torch.int32, device="cuda")

    g_infer_state_lock.acquire()
    mem_indexes = mem_manager.alloc(input_ids.shape[0]).cuda()
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": len(req_objs),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
    }

    return kwargs


# 双token
def prepare_mtp_main_model_decode_inputs(req_objs: List[Tuple], draft_token_id_map):
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

        # TODO: mtp_step > 1
        for step in range(1):
            run_reqs.append(req)
            nopad_b_req_idx.append(req.req_idx)
            seq_len = req.get_cur_total_len() + step + 1
            assert req.cur_kv_len == seq_len - step - 2
            nopad_b_seq_len.append(seq_len)
            input_ids.append(draft_token_id_map[req.req_idx])
            nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if g_infer_context.radix_cache is not None:
        g_infer_context.radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = g_infer_context.req_manager.mem_manager.alloc(input_ids.shape[0]).cuda()
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": len(run_reqs),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }
    return kwargs, run_reqs
