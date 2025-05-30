import torch
import triton
import triton.language as tl


@triton.jit
def _gen_sampling_params_kernel(
    req_to_presence_penalty,
    b_presence_penalty,
    req_to_frequency_penalty,
    b_frequency_penalty,
    req_to_repetition_penalty,
    b_repetition_penalty,
    req_to_temperature,
    b_temperature,
    req_to_exponential_decay_length_penalty,
    b_exponential_decay_length_penalty,
    req_to_req_idx,
    batch_size,
    BLOCK: tl.constexpr,  # num_warps
):

    block_start_index = tl.program_id(0) * BLOCK
    offs = block_start_index + tl.arange(0, BLOCK)
    mask = offs < batch_size

    req_idx = tl.load(req_to_req_idx + offs, mask=mask, other=0)

    tl.store(b_presence_penalty + offs, tl.load(req_to_presence_penalty + req_idx, mask=mask, other=0.0), mask=mask)
    tl.store(b_frequency_penalty + offs, tl.load(req_to_frequency_penalty + req_idx, mask=mask, other=0.0), mask=mask)
    tl.store(b_repetition_penalty + offs, tl.load(req_to_repetition_penalty + req_idx, mask=mask, other=0.0), mask=mask)
    tl.store(b_temperature + offs, tl.load(req_to_temperature + req_idx, mask=mask, other=0.0), mask=mask)
    tl.store(
        b_exponential_decay_length_penalty + offs,
        tl.load(req_to_exponential_decay_length_penalty + req_idx, mask=mask, other=0.0),
        mask=mask,
    )
    return


@torch.no_grad()
def gen_sampling_params(b_req_idx: torch.Tensor, req_sampling_params_manager):
    # fix circle import
    from lightllm.common.req_manager import ReqSamplingParamsManager

    req_sampling_params_manager: ReqSamplingParamsManager = req_sampling_params_manager

    batch_size = b_req_idx.shape[0]
    b_presence_penalty = torch.empty((batch_size,), dtype=torch.float32, device="cuda")
    b_frequency_penalty = torch.empty((batch_size,), dtype=torch.float32, device="cuda")
    b_repetition_penalty = torch.empty((batch_size,), dtype=torch.float32, device="cuda")
    b_temperature = torch.empty((batch_size,), dtype=torch.float32, device="cuda")
    b_exponential_decay_length_penalty = torch.empty((batch_size,), dtype=torch.float32, device="cuda")

    BLOCK = 256

    _gen_sampling_params_kernel[(triton.cdiv(batch_size, BLOCK),)](
        req_to_presence_penalty=req_sampling_params_manager.req_to_presence_penalty,
        b_presence_penalty=b_presence_penalty,
        req_to_frequency_penalty=req_sampling_params_manager.req_to_frequency_penalty,
        b_frequency_penalty=b_frequency_penalty,
        req_to_repetition_penalty=req_sampling_params_manager.req_to_repetition_penalty,
        b_repetition_penalty=b_repetition_penalty,
        req_to_temperature=req_sampling_params_manager.req_to_temperature,
        b_temperature=b_temperature,
        req_to_exponential_decay_length_penalty=req_sampling_params_manager.req_to_exponential_decay_length_penalty,
        b_exponential_decay_length_penalty=b_exponential_decay_length_penalty,
        batch_size=batch_size,
        BLOCK=BLOCK,
        num_warps=1,
    )
    return (
        b_presence_penalty,
        b_frequency_penalty,
        b_repetition_penalty,
        b_temperature,
        b_exponential_decay_length_penalty,
    )


@triton.jit
def _token_id_counter_kernel(
    prompt_ids_ptr,
    token_id_to_counter_ptr,
    input_size,
    vocab_size,
    BLOCK: tl.constexpr,
):

    block_start_index = tl.program_id(0) * BLOCK
    offs = block_start_index + tl.arange(0, BLOCK)
    mask = offs < input_size

    token_ids = tl.load(prompt_ids_ptr + offs, mask=mask, other=0)
    tl.atomic_add(token_id_to_counter_ptr + token_ids, 1, mask=(token_ids < vocab_size) & mask)
    return


@torch.no_grad()
def token_id_counter(prompt_ids: torch.Tensor, out_token_id_counter: torch.Tensor):
    vocab_size = out_token_id_counter.shape[0]
    input_size = prompt_ids.shape[0]
    BLOCK = 256

    _token_id_counter_kernel[(triton.cdiv(input_size, BLOCK),)](
        prompt_ids_ptr=prompt_ids,
        token_id_to_counter_ptr=out_token_id_counter,
        input_size=input_size,
        vocab_size=vocab_size,
        BLOCK=BLOCK,
        num_warps=1,
    )
    return
