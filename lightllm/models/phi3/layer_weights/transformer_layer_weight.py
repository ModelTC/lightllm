import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Phi3TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def load_hf_weights(self, weights):
        if f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight" in weights:
            qkv_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight"]
            n_embed = self.network_config_["hidden_size"]
            kv_n_embed = (
                n_embed // self.network_config_["num_attention_heads"] * self.network_config_["num_key_value_heads"]
            )
            q_weight_ = qkv_weight_[:n_embed, :]
            k_weight_ = qkv_weight_[n_embed : n_embed + kv_n_embed, :]
            v_weight_ = qkv_weight_[n_embed + kv_n_embed : n_embed + 2 * kv_n_embed, :]
            weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"] = q_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"] = k_weight_
            weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"] = v_weight_
            del weights[f"model.layers.{self.layer_num_}.self_attn.qkv_proj.weight"]

        if f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight" in weights:
            inter_size = self.network_config_["intermediate_size"]
            gate_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"]
            gate_weight_ = gate_up_proj[0:inter_size, :]
            up_weight_ = gate_up_proj[inter_size:, :]
            weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"] = gate_weight_
            weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"] = up_weight_
            del weights[f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"]
        super().load_hf_weights(weights)
