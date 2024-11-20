import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


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
        layer_prefix="model.layers",
    ):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)

        self.layer_name = f"{layer_prefix}.{layer_num}"

        self._init_config()
        self._init_weight_names()
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()
        self.set_quantization()
        return

    def _init_config(self):
        self.n_embed = self.network_config_["hidden_size"]
        self.n_head = self.network_config_["num_attention_heads"]
        self.n_inter = self.network_config_["intermediate_size"]
        self.n_kv_head = self.network_config_["num_key_value_heads"]
        self.head_dim = self.network_config_.get("head_dim", self.n_embed // self.n_head)

    def _init_weight_names(self):
        self._q_weight_name = f"{self.layer_name}.self_attn.q_proj.weight"
        self._q_bias_name = None
        self._k_weight_name = f"{self.layer_name}.self_attn.k_proj.weight"
        self._k_bias_name = None
        self._v_weight_name = f"{self.layer_name}.self_attn.v_proj.weight"
        self._v_bias_name = None
        self._o_weight_name = f"{self.layer_name}.self_attn.o_proj.weight"
        self._o_bias_name = None

        self._gate_weight_name = f"{self.layer_name}.mlp.gate_proj.weight"
        self._gate_bias_name = None
        self._up_weight_name = f"{self.layer_name}.mlp.up_proj.weight"
        self._up_bias_name = None
        self._down_weight_name = f"{self.layer_name}.mlp.down_proj.weight"
        self._down_bias_name = None

        self.att_norm_weight_name = f"{self.layer_name}.input_layernorm.weight"
        self.att_norm_bias_name = None
        self.ffn_norm_weight_name = f"{self.layer_name}.post_attention_layernorm.weight"
        self.ffn_norm_bias_name = None

    def _init_qkv(self):
        q_split_n_embed = self.head_dim * self.n_head // self.world_size_
        kv_split_n_embed = self.head_dim * self.n_kv_head // self.world_size_
        self.q_proj = ROWMMWeight(self._q_weight_name, self.data_type_, q_split_n_embed, bias_name=self._q_bias_name)
        self.k_proj = ROWMMWeight(
            self._k_weight_name, self.data_type_, kv_split_n_embed, bias_name=self._k_bias_name, wait_fuse=True
        )
        self.v_proj = ROWMMWeight(
            self._v_weight_name, self.data_type_, kv_split_n_embed, bias_name=self._v_bias_name, wait_fuse=True
        )

    def _init_o(self):
        o_split_n_embed = self.n_embed // self.world_size_
        self.o_proj = COLMMWeight(self._o_weight_name, self.data_type_, o_split_n_embed, bias_name=self._o_bias_name)

    def _init_ffn(self):
        split_inter_size = self.n_inter // self.world_size_
        self.gate_proj = ROWMMWeight(
            self._gate_weight_name, self.data_type_, split_inter_size, bias_name=self._gate_bias_name, wait_fuse=True
        )
        self.up_proj = ROWMMWeight(
            self._up_weight_name, self.data_type_, split_inter_size, bias_name=self._up_bias_name, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            self._down_weight_name, self.data_type_, split_inter_size, bias_name=self._down_bias_name
        )
        self.fuse_pairs.update({"gate_proj&up_proj": "gate_up_proj"})
        self.gate_up_proj: ROWMMWeight = None

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(
            self.att_norm_weight_name, self.data_type_, bias_name=self.att_norm_bias_name
        )
        self.ffn_norm_weight_ = NormWeight(
            self.ffn_norm_weight_name, self.data_type_, bias_name=self.ffn_norm_bias_name
        )
