import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        n_embed = self.network_config_["hidden_size"]
        # Dealing with head_dim_!=n_embed // num_attention_heads scenarios, such as mistral 13B
        head_dim = n_embed // self.network_config_["num_attention_heads"]
        self.head_dim = self.network_config_.get("head_dim", head_dim)
        self.fuse_pairs.update({"gate_proj&up_proj": "gate_up_proj"})

        self.init_qkv()
        self.init_o()
        self.init_ffn()
        self.init_norm()
        self.set_quantization()
        return

    def init_qkv(self):
        q_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        kv_split_n_embed = self.head_dim * self.network_config_["num_key_value_heads"] // self.world_size_
        self.q_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.q_proj.weight", self.data_type_, q_split_n_embed
        )
        self.k_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_proj.weight", self.data_type_, kv_split_n_embed, wait_fuse=True
        )
        self.v_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_proj.weight", self.data_type_, kv_split_n_embed, wait_fuse=True
        )

    def init_o(self):
        o_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        self.o_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight", self.data_type_, o_split_n_embed
        )

    def init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        self.gate_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.mlp.gate_proj.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.up_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.mlp.up_proj.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.mlp.down_proj.weight", self.data_type_, split_inter_size
        )

    def init_norm(self):
        self.att_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight", self.data_type_
        )
