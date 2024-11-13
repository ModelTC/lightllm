import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Internlm2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def load_hf_weights(self, weights):
        if f"model.layers.{self.layer_num_}.attention.wqkv.weight" in weights:
            qkv_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wqkv.weight"]
            q_groups = self.network_config_["num_attention_heads"] // self.network_config_["num_key_value_heads"]
            qkv_weight_ = qkv_weight_.reshape(
                self.network_config_["num_key_value_heads"], q_groups + 2, self.head_dim, -1
            )
            q_weight_ = qkv_weight_[:, :q_groups, :, :].reshape(-1, qkv_weight_.shape[-1])
            k_weight_ = qkv_weight_[:, -2, :, :].reshape(-1, qkv_weight_.shape[-1])
            v_weight_ = qkv_weight_[:, -1, :, :].reshape(-1, qkv_weight_.shape[-1])
            weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"] = q_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"] = k_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"] = v_weight_
            del weights[f"model.layers.{self.layer_num_}.attention.wqkv.weight"]
        super().load_hf_weights(weights)

    def init_o(self):
        o_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        self.o_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.attention.wo.weight", self.data_type_, o_split_n_embed
        )

    def init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        self.gate_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.feed_forward.w1.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.up_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.feed_forward.w3.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.feed_forward.w2.weight", self.data_type_, split_inter_size
        )

    def init_norm(self):
        self.att_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.attention_norm.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.ffn_norm.weight", self.data_type_)
