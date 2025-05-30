import torch
import time
import pytest
from lightllm.common.basemodel.triton_kernel.gen_sampling_params import token_id_counter, gen_sampling_params
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_token_id_counter():
    test_prompt_ids = torch.randint(0, 100, size=(7853,), device="cuda")
    test_token_id_counter = torch.zeros((100,), dtype=torch.int32, device="cuda")
    token_id_counter(prompt_ids=test_prompt_ids, out_token_id_counter=test_token_id_counter)

    true_out = torch.bincount(test_prompt_ids, minlength=test_token_id_counter.shape[0])
    assert torch.equal(test_token_id_counter, true_out)

    test_prompt_ids = torch.randint(0, 50 * 10000, size=(128 * 1024,), device="cuda")
    test_token_id_counter = torch.zeros((50 * 10000,), device="cuda", dtype=torch.int32)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(100):
        token_id_counter(prompt_ids=test_prompt_ids, out_token_id_counter=test_token_id_counter)
    end_event.record()
    logger.info(f"test_token_id_count cost time: {start_event.elapsed_time(end_event)} ms")


if __name__ == "__main__":
    pytest.main()
