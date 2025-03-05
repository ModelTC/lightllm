import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightTpl, MultiMMWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.utils.log_utils import init_logger
from typing import Dict, List, Optional

logger = init_logger(__name__)


class ROWMMWeight:
    def __new__(cls, **kwargs):
        quant_cfg = kwargs.pop("quant_cfg", None)
        layer_num_ = kwargs.pop("layer_num_", None)
        layer_name_ = kwargs.pop("layer_name", None)
        quant_method, quant_type = cls._get_quant_method(quant_cfg, layer_num_, layer_name_)
        kwargs["quant_method"] = quant_method
        if quant_type is None or not quant_method.quantized_weight:
            return UnquantizedROWMMWeight(**kwargs)
        elif quant_type == "fp8w8a8":
            pass
        else:
            raise ValueError(f"Unsupported quantization method: {quant_method}")

    def _get_quant_method(cls, quant_cfg: Quantcfg, layer_num_: int, layer_name: str) -> QuantizationMethod:
        quant_method = quant_cfg.get_quant_method(layer_num_, layer_name)
        quant_type = quant_cfg.get_quant_type(layer_num_, layer_name)
        logger.info(f"Layer {layer_num_} {layer_name} is set to {quant_type}")
        return quant_method, quant_type


class UnquantizedROWMMWeight(MMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.has_bias = bias_name is not None
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)

    def _slice_weight(self, weight: torch.Tensor):
        tp_size = weight.shape[0] // self.tp_world_size_
        return weight[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)

    def _slice_bias(self, bias):
        tp_size = bias.shape[0] // self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)


class UnquantizedMultiROWMMWeight(MultiMMWeightTpl):
    _slice_weight = UnquantizedROWMMWeight._slice_weight
    _slice_bias = UnquantizedROWMMWeight._slice_bias

    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)


class W8A8MultiROWMMWeight(MultiMMWeightTpl):
    # TODO: Implement this
    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)
