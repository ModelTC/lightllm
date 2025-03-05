import torch
from .mm_weight import (
    MMWeightTpl,
)
from lightllm.common.quantization import Quantcfg
from lightllm.utils.dist_utils import get_current_device_id
from typing import Dict, List, Optional


class ROWMMWeight(type):
    def __call__(cls, quant_cfg: Quantcfg = None, *args, **kwargs):

        if quant_cfg is None:
            return UnquantizedROWMMWeight(*args, **kwargs)
        else:
            raise ValueError(f"Unknown quant method: {quant_cfg.quant_type}")


class UnquantizedROWMMWeight(MMWeightTpl, metaclass=ROWMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        weight = None
        if self.weight_name in weights:
            weight = weights[self.weight_name]
            self.weight = weight[self.start : self.end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)[self.start : self.end]
            self.bias = bias.cuda(get_current_device_id())
        if weight is None:
            return
        self._post_load_weights()
        return


class ROWMMWeightNoTP(UnquantizedROWMMWeight):
    def __init__(
        self,
        weight_name: str,
        data_type: torch.dtype,
        split_n_embed: int,
        bias_name: Optional[str] = None,
    ) -> None:
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.start = 0
        self.end = split_n_embed
