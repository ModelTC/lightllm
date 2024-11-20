import torch
import math
import numpy as np

from lightllm.models.baichuan7b.layer_weights.transformer_layer_weight import BaiChuan7bTransformerLayerWeight
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight


class BaiChuan13bTransformerLayerWeight(BaiChuan7bTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def init_static_params(self):
        return BloomTransformerLayerWeight.init_static_params(self)
