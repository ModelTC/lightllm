from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    from vllm import _custom_ops as ops

    vllm_ops = ops
    HAS_VLLM = True
except:
    HAS_VLLM = False
    sgl_allreduce_ops = None
    logger.warning(
        "vllm is not installed, you can't use the api of it. \
                   You can solve it by running `pip install vllm`."
    )
