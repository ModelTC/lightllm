import os
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from functools import lru_cache
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import get_current_device_name

logger = init_logger(__name__)


class KernelConfigs(ABC):
    @classmethod
    def get_config_file_name(cls, params: Dict[str, Any]) -> str:
        json_str = json.dumps(params, sort_keys=True)
        json_str = json_str.replace(" ", "").replace("\n", "").replace('"', "")
        filename = json_str
        device_name = get_current_device_name().replace(" ", "_")
        return f"{filename}_{device_name}.json"

    @lru_cache(maxsize=None)
    def get_the_config(params: Dict[str, Any], config_dir_path) -> Optional[dict]:
        json_file_name = KernelConfigs.get_config_file_name(params)
        config_file_path = os.path.join(config_dir_path, "configs", json_file_name)

        if os.path.exists(config_file_path):
            return json.load(config_file_path)
        else:
            logger.warning(f"can not find config_path {config_file_path}")
            return None

    @classmethod
    def store_config(cls, params: Dict[str, Any], config_dir_path: str, dest_json: dict):
        json_file_name = KernelConfigs.get_config_file_name(params)
        config_file_path = os.path.join(config_dir_path, "configs", json_file_name)
        with open(config_file_path, mode="w") as file:
            json.dump(dest_json, file)
        return

    @classmethod
    @abstractmethod
    def try_to_get_best_config(cls, *args, **kwargs) -> dict:
        pass
