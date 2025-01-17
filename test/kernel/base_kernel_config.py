from lightllm.common.kernel_config import KernelConfigs
from frozendict import frozendict
from functools import lru_cache
from typing import Dict


class BaseKernelConfig(KernelConfigs):
    kernel_name: str = "basekernel"

    def closest_power_2(n: int) -> int:
        return 1 << (n - 1).bit_length() if n & (n - 1) else n

    @classmethod
    @lru_cache(maxsize=200)
    def try_to_get_best_config(
        cls,
        kernel_name,
        key_params: dict,  # 需要用frozendict包装
        search_keys: list,  # 需要用Frozenlist包装
    ) -> dict:
        cls.kernel_name = kernel_name

        finded_config = cls.get_the_config(key_params)

        if finded_config:
            config = finded_config
            for key in search_keys:
                config = config[min(config.keys(), key=lambda x: abs(int(x) - key))]
        else:
            config = None
        return config

    @classmethod
    def save_config(cls, kernel_name, key_params: dict, config_json: Dict[int, Dict[int, Dict]]):
        cls.kernel_name = kernel_name
        key_params = frozendict(key_params)
        return cls.store_config(key_params, config_json)
