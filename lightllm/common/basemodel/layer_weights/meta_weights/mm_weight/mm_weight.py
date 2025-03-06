import os
import torch
from abc import abstractmethod
from typing import Optional, Tuple, List, Dict, Union
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeightTpl
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def generate_scale_name(name, weight_scale_suffix, act_scale_suffix):
    weight_scale_name = None
    act_scale_name = None
    if weight_scale_suffix is not None:
        weight_scale_name = ".".join(name.split(".")[:-1] + [weight_scale_suffix])
    if act_scale_suffix is not None:
        act_scale_name = ".".join(name.split(".")[:-1] + [act_scale_suffix])
    return weight_scale_name, act_scale_name


class MMWeightTpl(BaseWeightTpl):
    def __init__(
        self,
        data_type: torch.dtype,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(tp_rank, tp_world_size)
        self.data_type_ = data_type
        self.quant_method = quant_method
        self.weight: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def mm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        if self.quant_method is not None:
            return self.quant_method.apply(
                input_tensor, self.weight, self.bias, out, use_custom_tensor_mananger=use_custom_tensor_mananger
            )
        if out is None:
            shape = (input_tensor.shape[0], self.weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=out)
        return torch.addmm(self.bias, input_tensor, self.weight, out=out)

    def verify_load(self) -> bool:
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.has_bias:
            load_ok = load_ok and self.bias is not None
        return load_ok

    def _process_weights(self, weight) -> None:
        if self.quant_method is not None:
            self.weight = self.quant_method.quantize(weight.to(self.data_type_).cuda(get_current_device_id()))
            return
        # 让 k dim 更连续，大多数split k 算法的算子可能能更快
        self.weight = weight.to(self.data_type_).cuda(get_current_device_id()).transpose(0, 1)

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:

        if self.weight_name in weights:
            weight = self._slice_weight(weights[self.weight_name])
            self._process_weights(weight)

        if self.bias_name in weights:
            self.bias = self._slice_bias(weights[self.bias_name]).cuda(get_current_device_id())
        return


class MultiMMWeightTpl(MMWeightTpl):
    def __init__(
        self,
        weight_names: str,
        data_type: torch.dtype,
        bias_names: Optional[str] = None,
        quant_method: QuantizationMethod = None,
        tp_rank: int = None,
        tp_world_size: int = None,
    ) -> None:
        super().__init__(data_type, quant_method, tp_rank, tp_world_size)

        self.weight_names = weight_names
        self.bias_names = bias_names
        self.weights = [None] * len(self.weight_names)
        self.biases = [None] * len(self.bias_names)
        self.has_bias = all(b is not None for b in self.bias_names) and len(bias_names) > 0

    def _fuse_weights(self) -> None:
        if self.weight is None and (None not in self.weights):
            weight = torch.cat(self.weights, dim=0)
            self._process_weights(weight)
            delattr(self, "weights")

        if self.has_bias and self.bias is None and (None not in self.biases):
            self.bias = torch.cat(self.biases, dim=0).cuda(get_current_device_id())
            delattr(self, "biases")
        return self

    def _load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        weight = None
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]]
                self.weights[i] = self._slice_weight(weight)
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]]
                self.biases[i] = self._slice_bias(bias)
        self._fuse_weights()


class MMWeight:
    def __new__(cls, **kwargs):
        quant_cfg = kwargs.pop("quant_cfg", None)
        layer_num_ = kwargs.pop("layer_num", None)
        layer_name_ = kwargs.pop("layer_name", None)
        quant_method, quantized_weight = cls._get_quant_method(quant_cfg, layer_num_, layer_name_)
        kwargs["quant_method"] = quant_method
        mmcls = cls._get_mmcls(quant_method, quantized_weight)
        return mmcls(**kwargs)

    @classmethod
    def _get_quant_method(cls, quant_cfg: Quantcfg, layer_num_: int, layer_name: str) -> QuantizationMethod:
        quant_method = quant_cfg.get_quant_method(layer_num_, layer_name)
        quant_type = quant_cfg.get_quant_type(layer_num_, layer_name)
        quantized_weight = quant_cfg.quantized_weight
        if quant_method is not None:
            logger.info(f"Layer {layer_num_} {layer_name} is set to {quant_type}")
        return quant_method, quantized_weight

    @classmethod
    def _get_mmcls(cls, quant_method: QuantizationMethod) -> Optional[Union[MMWeightTpl, MultiMMWeightTpl]]:
        return None


class BMMWeightTpl(MMWeightTpl):
    def __init__(self, data_type: torch.dtype):
        super().__init__(data_type)

    def set_quant_method(self, quant_method: QuantizationMethod) -> None:
        if self.quantized_weight:
            # for the quantized fp8 weight of Deepseek v3
            self.quant_method = quant_method

    def bmm(
        self, input_tensor: torch.Tensor, out: Optional[torch.Tensor] = None, use_custom_tensor_mananger: bool = True
    ) -> torch.Tensor:
        if self.quant_method is not None:
            fpweight = self.dequant_weight(self.weight[0], self.weight[1])
        else:
            fpweight = self.weight
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], fpweight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.bmm(input_tensor, fpweight, out=out)
        return torch.addbmm(self.bias, input_tensor, fpweight, out=out)

    def _post_load_weights(self) -> None:
        if self.quant_method is not None:
            if self.quantized_weight:
                if (
                    self.weight is not None
                    and self.weight_scale is not None
                    and (not self.static_activation or self.input_scale is not None)
                ):
                    if self.weight_scale.ndim > 1:
                        self.weight_scale = self.weight_scale.cuda(get_current_device_id())
                    self.weight = [self.weight.cuda(get_current_device_id()), self.weight_scale, self.input_scale]
            return
        self.weight = self.weight.cuda(get_current_device_id())


class BMMWeight(BMMWeightTpl):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(data_type)
        self.start = split_n_embed * self.tp_rank_
        self.end = split_n_embed * (self.tp_rank_ + 1)
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.weight_scale_name, self.act_scale_name = generate_scale_name(
            weight_name, weight_scale_suffix, act_scale_suffix
        )
        self.quantized_weight = self.weight_scale_name is not None
        self.static_activation = self.act_scale_name is not None

    def verify_load(self) -> None:
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class ROWBMMWeight(BMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, split_n_embed, bias_name, weight_scale_suffix, act_scale_suffix)

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

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        weight = None
        weight_scale = None
        input_scale = None
        if self.weight_name in weights:
            weight = weights[self.weight_name]
            self.weight = weight[self.start : self.end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)[self.start : self.end]
            self.bias = bias.cuda(get_current_device_id())

        if self.weight_scale_name is not None and self.weight_scale_name in weights:
            weight_scale = weights[self.weight_scale_name]
            # per channel or block-wise
            if weight_scale.shape[0] > 1:
                weight_scale = weight_scale.to(torch.float)[self.start : self.end]
            else:
                # per tensor
                weight_scale = weight_scale.to(torch.float)
            self.weight_scale = weight_scale

        if self.act_scale_name is not None and self.act_scale_name in weights:
            input_scale = weights[self.act_scale_name].to(torch.float)
            self.input_scale = input_scale.cuda(get_current_device_id())

        if weight is None and weight_scale is None and input_scale is None:
            return
        self._post_load_weights()
        return


class ROWBMMWeightNoTp(ROWBMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
        weight_scale_suffix: Optional[str] = None,
        act_scale_suffix: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, split_n_embed, bias_name, weight_scale_suffix, act_scale_suffix)
        self.start = 0
        self.end = split_n_embed
