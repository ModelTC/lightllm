import torch
import time
import pytest
import random
from lightllm.common.fused_moe.moe_silu_and_mul_mix_quant_ep import silu_and_mul_masked_fwd
from lightllm.common.fused_moe.moe_silu_and_mul import silu_and_mul_fwd
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@pytest.mark.parametrize(
    "expert_num, token_num, hidden_dim",
    [
        (
            expert_num,
            token_num,
            hidden_dim,
        )
        for expert_num in range(3, 6)
        for hidden_dim in [128, 256, 122, 258, 2048]
        for token_num in range(1, 7, 2)
    ],
)
def test_silu_and_mul_masked(expert_num, token_num, hidden_dim):
    in_tensor = torch.randn((expert_num, token_num, hidden_dim), dtype=torch.float16, device="cuda")
    out_tensor = torch.randn((expert_num, token_num, hidden_dim // 2), dtype=torch.float16, device="cuda")
    true_out_tensor = torch.randn((expert_num, token_num, hidden_dim // 2), dtype=torch.float16, device="cuda")
    masked_m = [random.randint(0, token_num) for _ in range(expert_num)]
    masked_m = torch.tensor(masked_m, dtype=torch.int32, device="cuda")

    silu_and_mul_fwd(in_tensor.view(-1, hidden_dim), true_out_tensor.view(-1, hidden_dim // 2))
    silu_and_mul_masked_fwd(in_tensor, out_tensor, masked_m)

    for expert_id, expert_token_num in enumerate(masked_m.cpu().numpy()):
        assert torch.allclose(
            true_out_tensor[expert_id, :expert_token_num, :],
            out_tensor[expert_id, :expert_token_num, :],
            atol=1e-3,
            rtol=1e-2,
        )
    return


if __name__ == "__main__":
    pytest.main()
