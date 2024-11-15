import torch
import math
import numpy as np
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class StarcoderTransformerLayerWeight(BloomTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg, layer_prefix="transformer.h"
        )
        assert network_config["num_attention_heads"] % self.world_size_ == 0

        return

    def _split_qkv_weight(self, weights):
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]

        qkv_weight_name = f"{self.layer_name}.attn.c_attn.weight"
        if qkv_weight_name in weights:
            qkv_weight = weights[qkv_weight_name]
            weights[f"{self._q_name}.weight"] = qkv_weight[:, :n_embed]
            weights[f"{self._k_name}.weight"] = qkv_weight[:, n_embed : n_embed + head_dim]
            weights[f"{self._v_name}.weight"] = qkv_weight[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_weight_name]

        qkv_bias_name = f"{self.layer_name}.attn.c_attn.bias"
        if qkv_bias_name in weights:
            qkv_bias = weights[qkv_bias_name]
            weights[f"{self._q_name}.bias"] = qkv_bias[:, :n_embed]
            weights[f"{self._k_name}.bias"] = qkv_bias[:, n_embed : n_embed + head_dim]
            weights[f"{self._v_name}.bias"] = qkv_bias[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_bias_name]

    def _init_name(self):
        self.network_config_["n_embed"] = self.network_config_["hidden_size"]
        super()._init_name(self)

        self.o_name = f"{self.layer_name}.attn.c_proj"
        self.up_proj_name = f"{self.layer_name}.mlp.c_fc"
        self.down_proj_name = f"{self.layer_name}.mlp.c_proj"
        self.att_norm_name = f"{self.layer_name}.ln_1"
        self.ffn_norm_name = f"{self.layer_name}.ln_2"
