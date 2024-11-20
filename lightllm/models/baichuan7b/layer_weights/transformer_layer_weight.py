import torch
import math
import numpy as np

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class BaiChuan7bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _init_config(self):
        self.network_config_["num_key_value_heads"] = self.network_config_["num_attention_heads"]
        super()._init_config()

    def load_hf_weights(self, weights):
        qkv_weight_name = f"{self.layer_name}.self_attn.W_pack.weight"
        if qkv_weight_name in weights:
            qkv_weights = weights[qkv_weight_name]
            split_size = qkv_weights.shape[0] // 3
            q_weights, k_weights, v_weights = torch.split(qkv_weights, split_size, dim=0)
            weights[self._q_weight_name] = q_weights
            weights[self._k_weight_name] = k_weights
            weights[self._v_weight_name] = v_weights
            del weights[qkv_weight_name]
        super().load_hf_weights(weights)
        return
