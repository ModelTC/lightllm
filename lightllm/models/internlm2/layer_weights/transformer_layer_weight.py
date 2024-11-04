import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight

from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Internlm2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_weight_,
            self.o_weight_,
            self.ffn_norm_weight_,
            self.gate_up_proj,
            self.down_proj,
        ]
        # handle internlm 20b, which has no bias, so set q k v o bias to zero
        if not self.network_config_.get("bias", True):
            for layer_type in ("q", "kv", "o"):
                attr_name = f"{layer_type}_bias_"
                if hasattr(self, attr_name):
                    continue
                setattr(self, attr_name, None)
        else:
            weights.append(self.q_bias_)
            weights.append(self.kv_bias_)
            weights.append(self.o_bias_)

        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.attention_norm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.attention_norm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )
        head_dim = n_embed // self.network_config_["num_attention_heads"]
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.attention.wqkv.weight" in weights:
            qkv_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wqkv.weight"]
            q_groups = self.network_config_["num_attention_heads"] // self.network_config_["num_key_value_heads"]
            qkv_weight_ = qkv_weight_.reshape(self.network_config_["num_key_value_heads"], q_groups + 2, head_dim, -1)
            q_weight_ = qkv_weight_[:, :q_groups, :, :].reshape(-1, qkv_weight_.shape[-1])
            self.q_weight_ = self.mm_op.preprocess_weight(
                q_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1) :]
            )

            k_weight_ = qkv_weight_[:, -2, :, :].reshape(-1, qkv_weight_.shape[-1])
            self.k_weight_ = k_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :]
            v_weight_ = qkv_weight_[:, -1, :, :].reshape(-1, qkv_weight_.shape[-1])
            self.v_weight_ = v_weight_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1) :]

        self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=0, handle_func=self.mm_op.preprocess_weight)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.attention.wo.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.attention.wo.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self.mm_op.preprocess_weight(self.o_weight_)
        if f"model.layers.{self.layer_num_}.attention.wo.bias" in weights:
            self.o_bias_ = weights[f"model.layers.{self.layer_num_}.attention.wo.bias"]
            self.o_bias_ = self._cuda(self.o_bias_) / self.world_size_
        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.ffn_norm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.ffn_norm.weight"])

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.feed_forward.w3.weight" in weights:
            self.up_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w3.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]

        if f"model.layers.{self.layer_num_}.feed_forward.w1.weight" in weights:
            self.gate_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w1.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]

        self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=0, handle_func=self.mm_op.preprocess_weight)

        if f"model.layers.{self.layer_num_}.feed_forward.w2.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.feed_forward.w2.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.down_proj = self.mm_op.preprocess_weight(self.down_proj)
        return
