import torch
import math
import numpy as np
from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


class QwenTransformerLayerWeight(InternlmTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)

    def load_hf_weights(self, weights):
        if f"transformer.h.{self.layer_num_}.attn.c_attn.weight" in weights:
            qkv_weight_ = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
            split_size = qkv_weight_.shape[0] // 3
            q_weight_, k_weight_, v_weight_ = torch.split(qkv_weight_, split_size, dim=0)
            weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"] = q_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"] = k_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"] = v_weight_
            del weights[f"transformer.h.{self.layer_num_}.attn.c_attn.weight"]
        if f"transformer.h.{self.layer_num_}.attn.c_attn.bias" in weights:
            qkv_bias = weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"]
            split_size = qkv_bias.shape[0] // 3
            q_bias_, k_bias_, v_bias_ = torch.split(qkv_bias, split_size, dim=0)
            weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.bias"] = q_bias_
            weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.bias"] = k_bias_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.bias"] = v_bias_
            del weights[f"transformer.h.{self.layer_num_}.attn.c_attn.bias"]
        super().load_hf_weights(weights)

    def init_o(self):
        o_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        self.o_proj = COLMMWeight(
            f"transformer.h.{self.layer_num_}.attn.c_proj.weight", self.data_type_, o_split_n_embed
        )

    def init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        self.gate_proj = ROWMMWeight(
            f"transformer.h.{self.layer_num_}.mlp.w2.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.up_proj = ROWMMWeight(
            f"transformer.h.{self.layer_num_}.mlp.w1.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            f"transformer.h.{self.layer_num_}.mlp.c_proj.weight", self.data_type_, split_inter_size
        )

    def init_norm(self):
        self.att_norm_weight_ = NormWeight(f"transformer.h.{self.layer_num_}.ln_1.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(f"transformer.h.{self.layer_num_}.ln_2.weight", self.data_type_)
