import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeight,
    MMWeightTpl,
    generate_scale_name,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.common.quantization.quantize_method import QuantizationMethod
from typing import Dict, List, Optional


class COLMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedCOLMMWeight
        else:
            return W8A8B128COLMMWeight


class UnquantizedCOLMMWeight(MMWeightTpl):
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

    def _slice_weight(self, tensor):
        tp_size = tensor.shape[1] // self.tp_world_size_
        return tensor[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)

    def _slice_bias(self, bias):
        tp_size = bias.shape[0] // self.tp_world_size_
        return (bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)] / self.tp_world_size_).to(self.data_type_)


class W8A8B128COLMMWeight(UnquantizedCOLMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(weight_name, data_type, bias_name, quant_method, tp_rank, tp_world_size)
        
        self.weight_scale_name, self.act_scale_name = generate_scale_name(
            weight_name, quant_method.weight_scale_suffix, quant_method.act_scale_suffix
        )
        self.weight_scale: Optional[torch.Tensor] = None
        self.block_size = self.quant_method.block_size
        self.quantized_weight = True

    def _slice_weight(self, tensor):
        tp_size = tensor.shape[1] // self.tp_world_size_
        return tensor[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        tp_size = weight_scale.shape[1] // self.tp_world_size_
        scale_start = tp_size * self.tp_rank_  
        scale_end = tp_size * (self.tp_rank_ + 1)
        return weight_scale[:, scale_start: scale_end].to(torch.float)

    def _process_weight_scale(self, weight_scale) -> None:
        self.weight_scale = weight_scale.transpose(0, 1).cuda(get_current_device_id())

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name in weights:
            weight_scale = self._slice_weight_scale(weights[self.weight_scale_name])
            self._process_weight_scale(weight_scale)
        if self.weight_name in weights and self.weight_scale is not None:
            self.weight = [
                self.weight,
                self.weight_scale,
                None,
            ]
        return
