import torch
import numpy as np
from typing import List, Tuple
from lightllm.server.router.model_infer.infer_batch import InferReq, g_infer_context
from lightllm.common.basemodel.infer_lock import g_infer_state_lock
from lightllm.common.basemodel.batch_objs import ModelInput

IS_NONE = -1


def prepare_mtp_prefill_inputs(
    req_objs: List[InferReq],
    model_input: ModelInput,
    last_hidden_states: torch.Tensor,
    draft_model_idx: int,
    is_chunked_mode: bool,
):
    input_ids = []
    for i, req in enumerate(req_objs):
        if is_chunked_mode:
            input_token_ids = req.get_chunked_input_token_ids_shift(draft_model_idx + 1)
        else:
            input_token_ids = req.get_input_token_ids_shift(draft_model_idx + 1)
        input_ids.append(input_token_ids[req.cur_kv_len :])
    input_ids = np.concatenate(input_ids, dtype=np.int64)
    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    model_input.input_ids = input_ids
    # mtp embedding
    model_input.hidden_states = last_hidden_states
    return model_input
