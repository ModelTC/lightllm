import torch
from .quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
import torch.nn.functional as F
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    from torchao.dtypes import to_affine_quantized_intx, AffineQuantizedTensor
    from torchao.dtypes import TensorCoreTiledLayoutType
    from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
    from torchao.quantization import (
        int4_weight_only,
        int8_weight_only,
        float8_weight_only,
        fpx_weight_only,
        int8_dynamic_activation_int8_weight,
        float8_dynamic_activation_float8_weight,
        quantize_,
    )
    from torchao.utils import (
        TORCH_VERSION_AT_LEAST_2_4,
        TORCH_VERSION_AT_LEAST_2_5,
    )
except:
    logger.warning("torchao is not installed, you can't use quant api of it")


class AOBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert TORCH_VERSION_AT_LEAST_2_4, "torchao requires torch >=2.4"
        self.quant_func = None

    def quantize(self, weight: torch.Tensor):
        """ """
        dummy_linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        dummy_linear.weight = torch.nn.Parameter(weight.cuda())
        quantize_(dummy_linear, self.quant_func)
        return dummy_linear.weight

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        pass


class AOW4A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self, group_size=128):
        super().__init__()
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"torchao int4-weightonly requires groupsize in [32,64,128,256], but gets {group_size}"
        self.group_size = group_size
        self.quant_func = int4_weight_only(group_size=self.group_size)

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        return F.linear(input_tensor, weights, bias)
        # return AffineQuantizedTensor._quantized_linear_op(input_tensor, weights, bias)


class AOW8A8QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_dynamic_activation_int8_weight()

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        return F.linear(input_tensor, weights, bias)


class AOW8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_weight_only()

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
        return F.linear(input_tensor, weights, bias)


class AOFP8W8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        assert is_cuda_8_9, "FP8 requires GPU with compute capability >= 8.9"
        self.quant_func = float8_weight_only()

    def apply(self, input_tensor, weights, bias=None, out=None, workspace=None):
        """ """
