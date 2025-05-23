import torch
import pytest
from lightllm.common.basemodel.triton_kernel.redundancy_topk_ids_repair import redundancy_topk_ids_repair
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def test_redundancy_topk_ids_repair():
    ep_expert_num = 4
    global_rank = 0
    redundancy_expert_num = 1
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    redundancy_expert_ids = torch.tensor(
        [
            0,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    redundancy_topk_ids_repair(
        topk_ids=topk_ids,
        redundancy_expert_ids=redundancy_expert_ids,
        ep_expert_num=ep_expert_num,
        global_rank=global_rank,
    )

    ans_topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ans_topk_ids = (ans_topk_ids // ep_expert_num) * redundancy_expert_num + ans_topk_ids
    new_redundancy_expert_ids = (redundancy_expert_ids // ep_expert_num) * redundancy_expert_num + redundancy_expert_ids
    ans_topk_ids[ans_topk_ids == new_redundancy_expert_ids[0]] = (
        (ep_expert_num + redundancy_expert_num) * global_rank + ep_expert_num + 0
    )

    assert torch.equal(topk_ids, ans_topk_ids)

    ep_expert_num = 4
    global_rank = 1
    redundancy_expert_num = 1
    topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )

    redundancy_expert_ids = torch.tensor(
        [
            5,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    redundancy_topk_ids_repair(
        topk_ids=topk_ids,
        redundancy_expert_ids=redundancy_expert_ids,
        ep_expert_num=ep_expert_num,
        global_rank=global_rank,
    )

    ans_topk_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [7, 9, 10, 11],
            [1, 3, 5, 7],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    ans_topk_ids = (ans_topk_ids // ep_expert_num) * redundancy_expert_num + ans_topk_ids
    new_redundancy_expert_ids = (redundancy_expert_ids // ep_expert_num) * redundancy_expert_num + redundancy_expert_ids
    ans_topk_ids[ans_topk_ids == new_redundancy_expert_ids[0]] = (
        (ep_expert_num + redundancy_expert_num) * global_rank + ep_expert_num + 0
    )

    assert torch.equal(topk_ids, ans_topk_ids)


if __name__ == "__main__":
    pytest.main()
