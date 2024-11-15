import torch
import math
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import BaseWeight, ROWMMWeight, COLMMWeight, NormWeight


class ChatGLM2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        assert network_config["num_attention_heads"] % self.world_size_ == 0

        self.layer_name = f"transformer.encoder.layers.{self.layer_num_}"

        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()
        self.set_quantization()
        return

    def _preprocess_weight(self, weights):
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        multi_query_group_num = self.network_config_["multi_query_group_num"]

        qkv_weight_name = f"{self.layer_name}.self_attention.query_key_value.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            weights[f"{self._q_name}.weight"] = qkv_weight_[:, :n_embed]
            weights[f"{self._k_name}.weight"] = qkv_weight_[:, n_embed : n_embed + head_dim * multi_query_group_num]
            weights[f"{self._v_name}.weight"] = qkv_weight_[
                :, n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim
            ]
            del weights[qkv_weight_name]

        qkv_bias_name = f"{self.layer_name}.self_attention.query_key_value.bias"
        if qkv_bias_name in weights:
            qkv_bias_ = weights[qkv_bias_name]
            weights[f"{self._q_name}.bias"] = qkv_bias_[:n_embed]
            weights[f"{self._k_name}.bias"] = qkv_bias_[:, n_embed : n_embed + head_dim * multi_query_group_num]
            weights[f"{self._v_name}.bias"] = qkv_bias_[
                :, n_embed + multi_query_group_num * head_dim : n_embed + 2 * multi_query_group_num * head_dim
            ]
            del weights[qkv_bias_name]

    def load_hf_weights(self, weights):
        self._preprocess_weight(weights)
        super().load_hf_weights(weights)

    def _init_qkv(self):
        n_embed = self.network_config_["hidden_size"]
        head_dim = self.network_config_["hidden_size"] // self.network_config_["num_attention_heads"]
        multi_query_group_num = self.network_config_["multi_query_group_num"]
        kv_split_n_embed = multi_query_group_num // self.world_size_ * head_dim
        q_split_n_embed = n_embed // self.world_size_

        self._q_name = f"{self.layer_name}.self_attention.q_proj"
        self._k_name = f"{self.layer_name}.self_attention.k_proj"
        self._v_name = f"{self.layer_name}.self_attention.v_proj"

        self.q_proj = ROWMMWeight(
            f"{self._q_name}.weight", self.data_type_, q_split_n_embed, bias_name=f"{self._q_name}.bias"
        )
        self.k_proj = ROWMMWeight(
            f"{self._k_name}.weight",
            self.data_type_,
            kv_split_n_embed,
            bias_name=f"{self._k_name}.bias",
            wait_fuse=True,
        )
        self.v_proj = ROWMMWeight(
            f"{self._v_name}.weight",
            self.data_type_,
            kv_split_n_embed,
            bias_name=f"{self._v_name}.bias",
            wait_fuse=True,
        )

    def _init_o(self):
        o_split_n_embed = self.network_config_["hidden_size"] // self.world_size_
        self._o_name = f"{self.layer_name}.self_attention.dense.weight"

        self.o_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight", self.data_type_, o_split_n_embed
        )

    def _init_ffn(self):
        ffn_hidden_size = self.network_config_["ffn_hidden_size"]
        split_inter_size = ffn_hidden_size // self.world_size_

        self.gate_up_proj = ROWMMWeight(
            f"{self.layer_name}.mlp.dense_h_to_4h.weight", self.data_type_, split_inter_size
        )
        self.down_proj = COLMMWeight(f"{self.layer_name}.mlp.dense_4h_to_h.weight", self.data_type_, split_inter_size)

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(f"{self.layer_name}.input_layernorm.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(f"{self.layer_name}.post_attention_layernorm.weight", self.data_type_)
