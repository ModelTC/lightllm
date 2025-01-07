import os
from functools import lru_cache
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=None)
def get_unique_server_name():
    service_uni_name = os.getenv("UNIQUE_SERVICE_NAME_ID")
    return service_uni_name
