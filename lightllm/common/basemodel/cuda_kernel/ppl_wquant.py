import torch

def quantize_int4_ppl(weight, group_size=128):
    """
    weight: [K, N]
    return:
        qweight: [K, N//8] int32 (packed int4*8) new pack_order
        q_scale: [K//group_size, N] int32
    """
    from lightllm_ppl_kernel import WeightPreProcess_i4
    qweight_new, q_scale = WeightPreProcess_i4(weight, group_size=group_size)
    # from atex.core.ffi import CUDA
    # qweight_new, q_scale = CUDA.WeightPreProcess_i4(weight, group_size=group_size)
    return qweight_new, q_scale

def matmul_dequantize_int4_ppl(
        x: torch.FloatTensor,
        qweight: torch.IntTensor,
        scale_weight: torch.IntTensor,
        group_size,
        workspace = None
) -> torch.FloatTensor:
    """
    x is activation:             (M, K) float16
    qweight is quant weight:     (N//8, K) int32 (int4*8 packed with pack_order)
    return tensor:               (M, N) float16
    """
    if workspace is None:
        workspace = torch.empty(size=[33554432], dtype=torch.int8, device='cuda') # 32MB workspace
    assert x.shape[1] == qweight.shape[1], "A must be a multiple of 8 in the last dimension"
    from lightllm_ppl_kernel import int4fp16_matmul
    return int4fp16_matmul(x, qweight, scale_weight, workspace, group_size)
    # from atex.core.ffi import CUDA
    # return CUDA.matmul_i4_fp16(x, qweight, scale_weight, workspace, group_size)