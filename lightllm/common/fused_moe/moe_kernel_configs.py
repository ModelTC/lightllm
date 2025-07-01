import os
from frozendict import frozendict
from functools import lru_cache
from lightllm.common.kernel_config import KernelConfigs
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class MoeGroupedGemmKernelConfig(KernelConfigs):
    kernel_name: str = "grouped_moe_gemm_kernel"

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
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config[min(finded_config.keys(), key=lambda x: abs(int(x) - M))]
            return config
        else:
            if M <= expert_num:
                config = {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 32,
                    "num_warps": 4,
                    "num_stages": 3,
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

    @classmethod
    def save_config(
        cls,
        N: int,
        K: int,
        topk_num: int,
        expert_num: int,
        mul_routed_weight: bool,
        use_fp8_w8a8: bool,
        out_dtype: str,
        config_json: dict,
    ):
        key_params = {
            "N": N,
            "K": K,
            "topk_num": topk_num,
            "expert_num": expert_num,
            "mul_routed_weight": mul_routed_weight,
            "use_fp8_w8a8": use_fp8_w8a8,
            "out_dtype": str(out_dtype),
        }
        key_params = frozendict(key_params)

        return cls.store_config(key_params, config_json)
