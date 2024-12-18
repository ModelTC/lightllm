import torch
from .better_fused_moe import moe_align, moe_align1


def test_moe_align():
    topk_ids = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 1, 4]], dtype=torch.int32, device="cuda")
    out = torch.zeros((5, 3), dtype=torch.int32, device="cuda")
    out.fill_(0)
    moe_align(topk_ids, out)
    true = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 1, 1], [0, 0, 1]], dtype=torch.int32, device="cuda")
    assert torch.equal(out, true)


def test_moe_align1():
    experts_info = torch.tensor(
        [
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 1, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    experts_token_num = torch.zeros((3,), dtype=torch.int32, device="cuda")

    moe_align1(experts_info, experts_token_num)

    true_experts_token_num = torch.tensor([3, 2, 3], device="cuda", dtype=torch.int32)
    true_experts_info = torch.tensor(
        [[0, 2, 4, 0, 1], [1, 3, 0, 1, 0], [0, 1, 3, 1, 0]], device="cuda:0", dtype=torch.int32
    )

    torch.equal(experts_token_num, true_experts_token_num)
    torch.equal(experts_info, true_experts_info)


if __name__ == "__main__":
    test_moe_align()
    test_moe_align1()
