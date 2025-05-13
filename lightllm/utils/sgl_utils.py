from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel
    import sgl_kernel.allreduce as sgl_allreduce_ops

    sgl_ops = sgl_kernel
    HAS_SGL_KERNEL = True
except:
    HAS_SGL_KERNEL = False
    sgl_allreduce_ops = None
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )
