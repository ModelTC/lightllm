import torch
import time
from .grouped_fused_moe import moe_align, moe_align1, grouped_matmul
from lightllm.utils.log_utils import init_logger

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger = init_logger(__name__)


def test_moe_align():
    expert_num = 5
    token_num = 3
    topk = 3
    topk_ids = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 1, 4]], dtype=torch.int32, device="cuda")
    out = torch.zeros((expert_num, token_num * topk), dtype=torch.int32, device="cuda")
    out.fill_(0)
    moe_align(topk_ids, out)
    true = torch.tensor(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(out, true)


def test_moe_align1():
    expert_num = 160
    topk_num = 6
    token_num = 200
    def get_one():
        rnd_logics = torch.randn(token_num, expert_num, device="cuda")
        topk_values, topk_ids = torch.topk(rnd_logics, topk_num, dim=1)
        
        experts_info = torch.zeros((expert_num, token_num * topk_num), dtype=torch.int32, device="cuda")
        experts_info.fill_(0)
        moe_align(topk_ids, experts_info)
        
        topk_weights = torch.randn((token_num, topk_num), dtype=torch.float32, device="cuda")
        experts_token_num = torch.zeros((expert_num,), dtype=torch.int32, device="cuda")
        experts_weights = torch.zeros(experts_info.shape, dtype=torch.float32, device="cuda")
        return experts_info, topk_weights, experts_weights, experts_token_num
    
    test_datas = [get_one() for _ in range(100)]

    moe_align1(*test_datas[0], topk_num)
    
    torch.cuda.synchronize()
    start = time.time()

    for i in range(60):
        moe_align1(*test_datas[i + 1], topk_num)
    torch.cuda.synchronize()

    print(f"cost time: {time.time() - start} s")


def test_grouped_matmul():
    test_dtype = torch.bfloat16
    token_inputs = torch.randn((10, 512), dtype=test_dtype, device="cuda") / 10
    experts_token_num = torch.tensor([1, 9], dtype=torch.int32, device="cuda")
    experts_to_token_index = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    experts_to_weights = torch.tensor(
        [
            [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    expert_weights = torch.randn((2, 1024, 512), dtype=test_dtype, device="cuda") / 10
    topk_num = 1
    out = torch.empty((10, 1024), dtype=test_dtype, device="cuda")
    # warm up
    grouped_matmul(
        token_inputs,
        None,
        experts_token_num,
        experts_to_token_index,
        experts_to_weights,
        expert_weights,
        None,
        topk_num,
        out,
        expert_token_limit=2 ** 31 - 1,
        mul_routed_weight=True,
        use_fp8_w8a8=False,
    )
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        grouped_matmul(
            token_inputs,
            None,
            experts_token_num,
            experts_to_token_index,
            experts_to_weights,
            expert_weights,
            None,
            topk_num,
            out,
            expert_token_limit=2 ** 31 - 1,
            mul_routed_weight=True,
            use_fp8_w8a8=False,
        )
    torch.cuda.synchronize()
    logger.info(f"grouped_matmul test cost time: {time.time() - start} s")

    ans_list = []
    ans_list.append(torch.matmul(token_inputs[0:1, :], expert_weights[0].transpose(0, 1)))
    for i in range(9):
        t_ans = torch.matmul(token_inputs[(i + 1) : (i + 2), :], expert_weights[1].transpose(0, 1))
        ans_list.append(t_ans)

    true_out = torch.cat(ans_list, dim=0)
    logger.info(f"grouped_matmul max delta {torch.max(torch.abs(out - 0.5*true_out))}")
    assert torch.allclose(0.5 * true_out, out, atol=1e-2, rtol=0)


if __name__ == "__main__":
    # test_moe_align()
    test_moe_align1()
    # test_grouped_matmul()
