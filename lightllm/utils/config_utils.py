import json
import os
from functools import lru_cache
from .envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def get_config_json(model_path: str):
    with open(os.path.join(model_path, "config.json"), "r") as file:
        json_obj = json.load(file)
    return json_obj


def get_eos_token_ids(model_path: str):
    config_json = get_config_json(model_path)
    try:
        eos_token_id = config_json["eos_token_id"]
    except:
        # for some multimode model.
        try:
            eos_token_id = config_json["llm_config"]["eos_token_id"]
        except:
            eos_token_id = config_json["text_config"]["eos_token_id"]

    if isinstance(eos_token_id, int):
        return [eos_token_id]
    if isinstance(eos_token_id, list):
        return eos_token_id
    assert False, "error eos_token_id format in config.json"


def get_model_architectures(model_path: str):
    try:
        config_json = get_config_json(model_path)
        arch = config_json["architectures"][0]
        return arch
    except:
        logger.error("can not get architectures from config.json, return unknown_architecture")
        return "unknown_architecture"


def get_vocab_size(model_path: str):
    try:
        config_json = get_config_json(model_path)
        vocab_size = config_json["vocab_size"]
        if not isinstance(vocab_size, int):
            vocab_size = int(vocab_size)
        return vocab_size
    except:
        logger.error("can not get vocab_size from config.json, return 0")
        return 0


def get_dtype(model_path: str):
    config_json = get_config_json(model_path)
    try:
        torch_dtype = config_json["torch_dtype"]
        return torch_dtype
    except:
        logger.warning("torch_dtype not in config.json, use float16 as default")
        return "float16"


@lru_cache(maxsize=None)
def get_fixed_kv_len():
    start_args = get_env_start_args()
    model_cfg = get_config_json(start_args.model_dir)
    if "prompt_cache_token_ids" in model_cfg:
        return len(model_cfg["prompt_cache_token_ids"])
    else:
        return 0
