from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

try:
    from vllm._custom_ops import *
except ImportError:
    try:
        from lightllm.common._ops import *
    except ImportError:
        logger.error("vllm or lightllm_kernel is not installed, you can't use custom ops")
