import torch
import math
import numpy as np
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import GEMMANormWeight, ROWMMWeight


class Gemma_2bTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def init_qkv(self):
        q_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        kv_split_n_embed = self.head_dim * self.network_config_["num_key_value_heads"]
        self.q_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.q_proj.weight", self.data_type_, q_split_n_embed
        )
        self.k_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            disable_tp=True,
        )
        self.v_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            disable_tp=True,
        )

    def init_norm(self):
        self.att_norm_weight_ = GEMMANormWeight(
            f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_
        )
        self.ffn_norm_weight_ = GEMMANormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight", self.data_type_
        )
