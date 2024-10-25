import torch
from .quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


class PPLW4A16QuantizationMethod(QuantizationMethod):
    def __init__(self, group_size=128):
        super().__init__()
        self.group_size = group_size

    def quantize(self, weight: torch.Tensor):
        """
        weight: [K, N]
        return:
            qweight: [K, N//8] int32 (packed int4*8) new pack_order
            q_scale: [K//group_size, N] int32
        """
        weight = weight.to(dtype=torch.float16).cuda()
        from lightllm_ppl_int4_kernel import int4_weight_encode

        qweight_new, q_scale = int4_weight_encode(weight, self.group_size)
        return qweight_new, q_scale

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """
        input_tensor is activation:             (M, K) float16
        weights: [qweight, scale_weight]
        qweight is quant weight:     (N//8, K) int32 (int4*8 packed with pack_order)
        return tensor:               (M, N) float16
        """
        qweight, scale_weight = weights
        if workspace is None:
            workspace = torch.empty(size=[33554432], dtype=torch.int8, device="cuda")  # 32MB workspace
            PPLW4A16QuantizationMethod.apply.__defaults__ = (None, None, workspace)
        if out is None:
            shape = (input_tensor.shape[0], qweight.shape[0] * 8)
            dtype = input_tensor.dtype
            device = input_tensor.device
            out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
        from lightllm_ppl_int4_kernel import matmul_i4_fp16
        from lightllm_ppl_int4_kernel import int4_weight_decode

        BATCHSIZE = input_tensor.shape[0]
        if BATCHSIZE >= 768:
            weight = int4_weight_decode(qweight, scale_weight, self.group_size)
            torch.mm(input_tensor, weight.transpose(0, 1), out=out)
        else:
            matmul_i4_fp16(input_tensor, qweight, scale_weight, workspace, self.group_size, out)
        if bias is None:
            return out
        else:
            out.add_(bias)
            return out
