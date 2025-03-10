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
        self.weight_tp_size = tp_size
        return tensor[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)].to(self.data_type_)


class W8A8B128COLMMWeight(UnquantizedCOLMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        bias_name: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, bias_name, quant_method, tp_rank, tp_world_size)
        self.weight_scale_name, self.act_scale_name = generate_scale_name(
            weight_name, weight_scale_suffix, act_scale_suffix
        )
        self.weight_scale: Optional[torch.Tensor] = None
        self.input_scale: Optional[torch.Tensor] = None
        self.quantized_weight = self.weight_scale_name is not None
        self.static_activation = self.act_scale_name is not None
        self.block_size = self.quant_method.block_size

    def _slice_weight(self, tensor):
        tp_size = tensor.shape[1] // self.tp_world_size_
        self.weight_tp_size = tp_size
        return tensor[:, tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        scale_start = (self.weight_tp_size * self.tp_rank_) // self.block_size
        scale_end = (self.weight_tp_size * (self.tp_rank_ + 1)) // self.block_size
        return weight_scale[:, scale_start: scale_end].to(torch.float)

    def _post_process_weight_scale(self, weight_scale) -> None:
        if weight_scale.ndim > 1:
            self.weight_scale = weight_scale.transpose(0, 1).cuda(get_current_device_id())
        else:
            self.weight_scale = weight_scale

    def _post_process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _load_scales(self, weights: Dict[str, torch.Tensor]) -> None:
        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            # per channel or block-wise
            if weight_scale.shape[0] > 1:
                weight_scale = self._slice_weight_scale(weight_scale)
            else:
                # per tensor
                weight_scale = weight_scale.to(torch.float)
            self._post_process_weight_scale(weight_scale)

        if self.static_activation and self.act_scale_name is not None and self.act_scale_name in weights:
            input_scale = weights[self.act_scale_name].to(torch.float)
            self.input_scale = input_scale.cuda(get_current_device_id())

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                self.input_scale,
            ]
        return
