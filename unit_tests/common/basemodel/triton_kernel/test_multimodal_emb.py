import torch
import pytest
from lightllm.common.basemodel.triton_kernel.multimodal_emb import mark_multimodal_obj
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_mark_mubltimodal_obj():
    obj_start_ids = torch.tensor([1, 4, 100], device="cuda", dtype=torch.int64)
    obj_token_lens = torch.tensor([1, 3, 2], device="cuda", dtype=torch.int64)
    input_ids = torch.tensor([1, 7, 9, 333], device="cuda", dtype=torch.int64)

    mark_obj = mark_multimodal_obj(
        obj_start_token_ids=obj_start_ids, obj_token_lens=obj_token_lens, input_ids=input_ids
    )

    assert torch.equal(mark_obj, torch.tensor([1, 0, 0], device="cuda"))


if __name__ == "__main__":
    pytest.main()
