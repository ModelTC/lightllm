import torch
import math
import numpy as np

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class BaiChuan7bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def load_hf_weights(self, weights):
        self.network_config_["num_key_value_heads"] = self.network_config_["num_attention_heads"]
        if f"model.layers.{self.layer_num_}.self_attn.W_pack.weight" in weights:
            qkv_weights = weights[f"model.layers.{self.layer_num_}.self_attn.W_pack.weight"]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)
            weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"] = q_weights
            weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"] = k_weights
            weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"] = v_weights
            del weights[f"model.layers.{self.layer_num_}.self_attn.W_pack.weight"]
        super().load_hf_weights(weights)
        return
