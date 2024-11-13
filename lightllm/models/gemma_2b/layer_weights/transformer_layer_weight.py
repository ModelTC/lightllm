import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import GEMMANormWeight


class Gemma_2bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def init_norm(self):
        self.att_norm_weight_ = GEMMANormWeight(
            f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_
        )
        self.ffn_norm_weight_ = GEMMANormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight", self.data_type_
        )
