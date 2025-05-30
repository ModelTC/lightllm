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


if __name__ == "__main__":
    pytest.main()
