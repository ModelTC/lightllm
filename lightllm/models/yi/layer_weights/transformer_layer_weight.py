import torch
import math
import numpy as np

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NormWeight


class YiTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight_names(self):
        super()._init_weight_names()
        self.att_norm_weight_name = f"{self.layer_name}.ln1.weight"
        self.ffn_norm_weight_name = f"{self.layer_name}.ln2.weight"
