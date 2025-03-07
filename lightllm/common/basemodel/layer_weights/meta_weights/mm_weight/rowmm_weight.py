import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeight,
    MMWeightTpl,
    BMMWeightTpl,
    MultiMMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional


class ROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWMMWeight
        # TODO: Implement more quantization weight
        return None


class MultiROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedMultiROWMMWeight
        # TODO: Implement more quantization weight
        return None


class ROWBMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWBMMWeight
        # TODO: Implement more quantization weight
        return None


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


class UnquantizedROWBMMWeight(BMMWeightTpl):
    _slice_weight = UnquantizedROWMMWeight._slice_weight
    _slice_bias = UnquantizedROWMMWeight._slice_bias

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

    def dequant_weight(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # for Deepseek v3
        # TODO a fast bmm quant kernel
        weight = weight.to(self.data_type_)
        block_size = weight.shape[-1] // scale.shape[-1]
        w_shape = weight.shape
        s_shape = scale.shape
        scale = scale.unsqueeze(-1).repeat(1, 1, 1, block_size).reshape(s_shape[0], s_shape[1], -1)
        scale = scale.unsqueeze(2).repeat(1, 1, block_size, 1).reshape(w_shape)
        return (weight * scale).to(self.data_type_)


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
