import torch
import math
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class MiniCPMTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        num_hidden_layers = self.network_config_["num_hidden_layers"]
        scale_depth = self.network_config_.get("scale_depth", math.sqrt(num_hidden_layers))
        self.layer_scale = scale_depth / math.sqrt(num_hidden_layers)
        return

    def load_hf_weights(self, weights):
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"] *= self.layer_scale
        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"] *= self.layer_scale
        super().load_hf_weights(weights)
