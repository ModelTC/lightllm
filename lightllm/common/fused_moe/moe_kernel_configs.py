import os
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MoeGroupedGemmKernelConfig(KernelConfigs):
    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        M: int,
        N: int,
        K: int,
        topk_num: int,
        expert_num: int,
        mul_routed_weight: bool,
        use_fp8_w8a8: bool,
        out_dtype: str,
    ) -> dict:
        key_params = {
            "N": N,
            "K": K,
            "topk_num": topk_num,
            "expert_num": expert_num,
            "mul_routed_weight": mul_routed_weight,
            "use_fp8_w8a8": use_fp8_w8a8,
            "out_dtype": out_dtype,
        }
        key_params = frozendict(key_params)

        finded_config = KernelConfigs.get_the_config(key_params, os.path.dirname(os.path.realpath(__file__)))

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(x - M))]
            return config
        else:
            if M <= expert_num:
                config = {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 1,
                }
            else:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                    "num_warps": 4,
                    "num_stages": 1,
                }
        return config
