import torch
from .quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
import torch.nn.functional as F

try:
    HAS_VLLM = True
    import vllm
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import apply_int8_linear
except:
    HAS_VLLM = False


class vLLMBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_VLLM, "vllm is not installed, you can't use quant api of it"

    def quantize(self, weight: torch.Tensor):
        """ """
        pass

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        pass


class vLLMw8a8QuantizationMethod(vLLMBaseQuantizationMethod):
    def __init__(self):
        super().__init__()

    def quantize(self, weight: torch.Tensor):
        if hasattr(weight, "scale"):
            return weight.data.transpose(0, 1).cuda(), weight.scale.cuda()
        weight = weight.float()
        scale = weight.abs().max(dim=-1)[0] / 127
        weight = weight.transpose(0, 1) / scale.reshape(1, -1)
        weight = torch.round(weight.clamp(min=-128, max=127)).to(dtype=torch.int8)
        return weight.cuda(), scale.cuda()

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        return apply_int8_linear(input_tensor, weights[0], weights[1], bias=bias)
