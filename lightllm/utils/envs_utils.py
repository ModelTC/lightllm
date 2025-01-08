import os
import json
from easydict import EasyDict
from functools import lru_cache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_unique_server_name():
    service_uni_name = os.getenv("UNIQUE_SERVICE_NAME_ID")
    return service_uni_name


def set_env_start_args(args):
    args_dict = vars(args)
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(args_dict)
    return


@lru_cache(maxsize=None)
def get_env_start_args():
    start_args = json.loads(os.environ["LIGHTLLM_START_ARGS"])
    start_args = EasyDict(start_args)
    return start_args
