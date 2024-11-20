import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class StarcoderTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg, layer_prefix="transformer.h"
        )
        assert network_config["num_attention_heads"] % self.world_size_ == 0

    def load_hf_weights(self, weights):
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]

        qkv_weight_name = f"{self.layer_name}.attn.c_attn.weight"
        if qkv_weight_name in weights:
            qkv_weight = weights[qkv_weight_name]
            weights[self._q_weight_name] = qkv_weight[:, :n_embed]
            weights[self._k_weight_name] = qkv_weight[:, n_embed : n_embed + head_dim]
            weights[self._v_weight_name] = qkv_weight[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_weight_name]

        qkv_bias_name = f"{self.layer_name}.attn.c_attn.bias"
        if qkv_bias_name in weights:
            qkv_bias = weights[qkv_bias_name]
            weights[self._q_bias_name] = qkv_bias[:, :n_embed]
            weights[self._k_bias_name] = qkv_bias[:, n_embed : n_embed + head_dim]
            weights[self._v_bias_name] = qkv_bias[:, n_embed + head_dim : n_embed + 2 * head_dim]
            del weights[qkv_bias_name]
        super().load_hf_weights(weights)

    def _init_weight_names(self):
        super()._init_weight_names(self)
        self._q_bias_name = f"{self.layer_name}.self_attn.q_proj.bias"
        self._k_bias_name = f"{self.layer_name}.self_attn.k_proj.bias"
        self._v_bias_name = f"{self.layer_name}.self_attn.v_proj.bias"

        self._o_weight_name = f"{self.layer_name}.attn.c_proj.weight"
        self._o_bias_name = f"{self.layer_name}.attn.c_proj.bias"

        self._up_weight_name = f"{self.layer_name}.mlp.c_fc.weight"
        self._up_bias_name = f"{self.layer_name}.mlp.c_fc.bias"
        self._down_weight_name = f"{self.layer_name}.mlp.c_proj.weight"
        self._down_bias_name = f"{self.layer_name}.mlp.c_proj.bias"

        self.att_norm_weight_name = f"{self.layer_name}.ln_1.weight"
        self.att_norm_bias_name = f"{self.layer_name}.ln_1.bias"
        self.ffn_norm_weight_name = f"{self.layer_name}.ln_2.weight"
        self.ffn_norm_bias_name = f"{self.layer_name}.ln_2.bias"

    def _init_ffn(self):
        split_inter_size = self.n_inter // self.world_size_
        self.up_proj = ROWMMWeight(
            self._up_weight_name, self.data_type_, split_inter_size, bias_name=self._up_bias_name, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            self._down_weight_name, self.data_type_, split_inter_size, bias_name=self._down_bias_name
        )
