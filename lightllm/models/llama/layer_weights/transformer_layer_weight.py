import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight, MultiROWMMWeight


class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(
        self,
        layer_num,
        tp_rank,
        world_size,
        data_type,
        network_config,
        mode=[],
        quant_cfg=None,
    ):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()

    def load_hf_weights(self, weights):
        self.k_weight = weights.get(self._k_weight_name, None)
        self.v_weight = weights.get(self._v_weight_name, None)
        self._gate_weight = weights.get(self._gate_weight_name, None)
        self._up_weight = weights.get(self._up_weight_name, None)
        kv_weight = self._try_cat_to(["k_weight", "v_weight"], 0)
        if kv_weight is not None:
            weights[self._kv_weight_name] = kv_weight

        gate_up_weight = self._try_cat_to(
            [
                "_gate_weight",
                "_up_weight",
            ],
            0,
        )
        if gate_up_weight is not None:
            weights[self._gate_up_weight_name] = gate_up_weight
        return super().load_hf_weights(weights)

    def _parse_config(self):
        self.n_embed = self.network_config_["hidden_size"]
        self.n_head = self.network_config_["num_attention_heads"]
        self.n_inter = self.network_config_["intermediate_size"]
        self.n_kv_head = self.network_config_["num_key_value_heads"]
        self.head_dim = self.network_config_.get("head_dim", self.n_embed // self.n_head)

    def _init_weight_names(self):
        self._q_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._kv_weight_name = f"model.layers.{self.layer_num_}.self_attn.kv_proj.weight"
        self._kv_bias_name = None
        self._o_weight_name = f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"
        self._o_bias_name = None

        self._gate_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"model.layers.{self.layer_num_}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._gate_up_weight_name = f"model.layers.{self.layer_num_}.mlp.gate_up_proj.weight"
        self._gate_up_bias_name = None
        self._down_weight_name = f"model.layers.{self.layer_num_}.mlp.down_proj.weight"
        self._down_bias_name = None

        self._att_norm_weight_name = f"model.layers.{self.layer_num_}.input_layernorm.weight"
        self._att_norm_bias_name = None
        self._ffn_norm_weight_name = f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"
        self._ffn_norm_bias_name = None

    def _init_qkv(self):
        kv_split_n_embed = self.head_dim * self.n_kv_head // self.world_size_
        self.q_proj = ROWMMWeight(self._q_weight_name, self.data_type_, bias_name=self._q_bias_name)
        self.kv_proj = ROWMMWeight(
            self._kv_weight_name,
            self.data_type_,
            kv_split_n_embed,
            bias_names=[self._k_bias_name, self._v_bias_name],
        )

    def _init_o(self):
        o_split_n_embed = self.head_dim * self.n_head // self.world_size_
        self.o_proj = COLMMWeight(self._o_weight_name, self.data_type_, o_split_n_embed, bias_name=self._o_bias_name)

    def _init_ffn(self):
        split_inter_size = self.n_inter // self.world_size_
        self.gate_up_proj = MultiROWMMWeight(
            [self._gate_weight_name, self._up_weight_name],
            self.data_type_,
            split_inter_size,
            bias_names=[self._gate_bias_name, self._up_bias_name],
        )
        self.down_proj = COLMMWeight(
            self._down_weight_name, self.data_type_, split_inter_size, bias_name=self._down_bias_name
        )

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(
            self._att_norm_weight_name, self.data_type_, bias_name=self._att_norm_bias_name
        )
        self.ffn_norm_weight_ = NormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
