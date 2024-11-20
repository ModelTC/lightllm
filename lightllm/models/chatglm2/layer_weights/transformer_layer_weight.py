import torch
import math
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight


class ChatGLM2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num,
            tp_rank,
            world_size,
            data_type,
            network_config,
            mode,
            quant_cfg,
            layer_prefix="transformer.encoder.layers",
        )
        return

    def _preprocess_weight(self, weights):
        n_kv_embed = self.head_dim * self.n_kv_head

        qkv_weight_name = f"{self.layer_name}.self_attention.query_key_value.weight"
        if qkv_weight_name in weights:
            qkv_weight_ = weights[qkv_weight_name]
            weights[self._q_weight_name] = qkv_weight_[: self.n_embed, :]
            weights[self._k_weight_name] = qkv_weight_[self.n_embed : self.n_embed + n_kv_embed, :]
            weights[self._v_weight_name] = qkv_weight_[self.n_embed + n_kv_embed : self.n_embed + 2 * n_kv_embed, :]
            del weights[qkv_weight_name]

        qkv_bias_name = f"{self.layer_name}.self_attention.query_key_value.bias"
        if qkv_bias_name in weights:
            qkv_bias_ = weights[qkv_bias_name]
            weights[self._q_bias_name] = qkv_bias_[: self.n_embed]
            weights[self._k_bias_name] = qkv_bias_[self.n_embed : self.n_embed + n_kv_embed]
            weights[self._v_bias_name] = qkv_bias_[self.n_embed + n_kv_embed : self.n_embed + 2 * n_kv_embed]
            del weights[qkv_bias_name]

    def _init_config(self):
        self.n_embed = self.network_config_["hidden_size"]
        self.n_head = self.network_config_["num_attention_heads"]
        self.n_inter = self.network_config_["ffn_hidden_size"]
        self.n_kv_head = self.network_config_["multi_query_group_num"]
        self.head_dim = self.network_config_.get("head_dim", self.n_embed // self.n_head)

    def load_hf_weights(self, weights):
        self._preprocess_weight(weights)
        super().load_hf_weights(weights)
        return

    def _init_ffn(self):
        split_inter_size = self.n_inter // self.world_size_
        self.up_proj = ROWMMWeight(
            self._up_weight_name, self.data_type_, split_inter_size, bias_name=self._up_bias_name, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            self._down_weight_name, self.data_type_, split_inter_size, bias_name=self._down_bias_name
        )
