from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel

    sgl_ops = sgl_kernel
    sgl_allreduce_ops = sgl_ops.allreduce
    if sgl_ops.flash_attn.is_fa3_supported():
        flash_attn_varlen_func = sgl_ops.flash_attn.flash_attn_varlen_func
        flash_attn_with_kvcache = sgl_ops.flash_attn.flash_attn_with_kvcache
        merge_state_v2 = sgl_ops.flash_attn.merge_state_v2
    else:
        flash_attn_varlen_func = None
        flash_attn_with_kvcache = None
        merge_state_v2 = None
        logger.warning("Fa3 is only supported on sm90 and above.")
    HAS_SGL_KERNEL = True
except:
    HAS_SGL_KERNEL = False
    sgl_allreduce_ops = None
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )
