import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight


class Qwen2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)

    def init_qkv(self):
        q_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        kv_split_n_embed = self.head_dim * self.network_config_["num_key_value_heads"] // self.world_size_
        self.q_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
            self.data_type_,
            q_split_n_embed,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.q_proj.bias",
        )
        self.k_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.k_proj.bias",
        )
        self.v_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.v_proj.bias",
        )
