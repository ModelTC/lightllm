import torch
from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import (
    MMWeight,
    MMWeightTpl,
    BMMWeightTpl,
    MultiMMWeightTpl,
    generate_scale_name,
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
        else:
            return W8A8B128ROWMMWeight
        # TODO: Implement more quantization weight
        return None


class MultiROWMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedMultiROWMMWeight
        else:
            return W8A8B128MultiROWMMWeight
        # TODO: Implement more quantization weight
        return None


class ROWBMMWeight(MMWeight):
    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod, quantized_weight: bool):
        if quant_method is None or not quantized_weight:
            return UnquantizedROWBMMWeight
        else:
            return W8A8B128ROWBMMWeight
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


class W8A8B128ROWMMWeight(UnquantizedROWMMWeight):
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

    def _slice_weight(self, weight: torch.Tensor):
        tp_size = weight.shape[0] // self.tp_world_size_
        self.weight_tp_size = tp_size
        return weight[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_bias(self, bias):
        tp_size = bias.shape[0] // self.tp_world_size_
        return bias[tp_size * self.tp_rank_ : tp_size * (self.tp_rank_ + 1)]

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        scale_start = (self.weight_tp_size * self.tp_rank_ + self.block_size - 1) // self.block_size
        scale_end = (self.weight_tp_size * (self.tp_rank_ + 1) + self.block_size - 1) // self.block_size
        return weight_scale.to(torch.float)[scale_start : scale_end]

    def _post_process_weight_scale(self, weight_scale) -> None:
        if weight_scale.ndim > 1:
            self.weight_scale = weight_scale.cuda(get_current_device_id()).transpose(0, 1)
        else:
            self.weight_scale = weight_scale

    def _post_process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        super()._load_weights(weights)
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


class W8A8B128MultiROWMMWeight(UnquantizedMultiROWMMWeight):
    _slice_weight = W8A8B128ROWMMWeight._slice_weight
    _slice_bias = W8A8B128ROWMMWeight._slice_bias

    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(weight_names, data_type, bias_names, quant_method, tp_rank, tp_world_size)
        self.weight_scale_names = []
        self.act_scale_names = []
        self.weight_scale: Optional[torch.Tensor] = None
        self.input_scale: Optional[torch.Tensor] = None
        self.block_size = self.quant_method.block_size
        self.input_scales = [None] * len(self.weight_names)
        self.weight_scales = [None] * len(self.weight_names)
        for weight_name in weight_names:
            weight_scale_name, act_scale_name = generate_scale_name(weight_name, weight_scale_suffix, act_scale_suffix)
            self.weight_scale_names.append(weight_scale_name)
            self.act_scale_names.append(act_scale_name)
            self.quantized_weight = weight_scale_name is not None
            self.static_activation = act_scale_name is not None

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        scale_start = (self.weight_tp_size * self.tp_rank_ + self.block_size - 1) // self.block_size
        scale_end = (self.weight_tp_size * (self.tp_rank_ + 1) + self.block_size - 1) // self.block_size
        return weight_scale[scale_start : scale_end].to(torch.float)

    def _pre_porcess_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        super()._pre_porcess_weights(weights)
        for i in range(len(self.weight_names)):
            if self.weight_scale_names[i] in weights:
                weight_scale = weights[self.weight_scale_names[i]]
                # block-wise or per-channel
                if weight_scale.shape[0] > 1:
                    weight_scale = self._slice_weight_scale(weight_scale)
                else:
                    # per tensor
                    weight_scale = weight_scale.to(torch.float)
                self.weight_scales[i] = weight_scale

            if self.static_activation and self.act_scale_names[i] in weights:
                input_scale = weights[self.act_scale_names[i]].to(torch.float)
                self.input_scales[i] = input_scale

    def _post_process_weight_scale(self, weight_scale) -> None:
        if weight_scale.ndim > 1:
            self.weight_scale = weight_scale.transpose(0, 1).cuda(get_current_device_id())
        else:
            self.weight_scale = weight_scale

    def _post_process_weight(self, weight) -> None:
        self.weight = weight.cuda(get_current_device_id()).transpose(0, 1)

    def _fuse_weights(self) -> None:
        super()._fuse_weights()
        if self.weight_scale is None and (None not in self.weight_scales):
            weight_scale = torch.cat(self.weight_scales, dim=0).cuda(get_current_device_id())
            self._post_process_weight_scale(weight_scale)
            delattr(self, "weight_scales")

        if self.static_activation and self.input_scale is None and (None not in self.input_scales):
            input_scales = torch.stack(self.input_scales, dim=0)
            self.input_scale = torch.max(input_scales).cuda(get_current_device_id())
            delattr(self, "input_scales")

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                self.input_scale,
            ]


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


class W8A8B128ROWBMMWeight(UnquantizedROWBMMWeight):
    _slice_weight = W8A8B128ROWMMWeight._slice_weight
    _slice_bias = W8A8B128ROWMMWeight._slice_bias

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
        # self.block_size = self.quant_method.block_size

    def _slice_weight_scale(self, weight_scale: torch.Tensor):
        scale_start = self.weight_tp_size * self.tp_rank_
        scale_end = self.weight_tp_size * (self.tp_rank_ + 1)
        return weight_scale[scale_start : scale_end].to(torch.float)

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        super()._load_weights(weights)
        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            # per channel or block-wise
            if weight_scale.shape[0] > 1:
                weight_scale = self._slice_weight_scale(weight_scale)
            else:
                # per tensor
                weight_scale = weight_scale.to(torch.float)
            self.weight_scale = weight_scale.cuda(get_current_device_id())

        if self.static_activation and self.act_scale_name is not None and self.act_scale_name in weights:
            input_scale = weights[self.act_scale_name].to(torch.float)
            self.input_scale = input_scale.cuda(get_current_device_id())

        if self.weight_scale is not None and isinstance(self.weight, torch.Tensor):
            self.weight = [
                self.weight,
                self.weight_scale,
                self.input_scale,
            ]
