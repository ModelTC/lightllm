from lightllm.common.basemodel.layer_weights.meta_weights import NormWeight, ROWMMWeight, COLMMWeight
from lightllm.common.basemodel import TransformerLayerWeight


class Starcoder2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        assert network_config["num_attention_heads"] % self.world_size_ == 0

    def init_norm(self):
        self.att_norm_weight_ = NormWeight(
            f"model.layers.{self.layer_num_}.input_layernorm.weight",
            self.data_type_,
            bias_name=f"model.layers.{self.layer_num_}.input_layernorm.bias",
        )
        self.ffn_norm_weight_ = NormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight",
            self.data_type_,
            bias_name=f"model.layers.{self.layer_num_}.post_attention_layernorm.bias",
        )

    def init_qkv(self):
        q_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        kv_split_n_embed = self.head_dim * self.network_config_["num_key_value_heads"] // self.world_size_
        self.q_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
            self.data_type_,
            q_split_n_embed,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.q_proj.bias",
        )
        self.k_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.k_proj.bias",
        )
        self.v_proj = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_proj.weight",
            self.data_type_,
            kv_split_n_embed,
            wait_fuse=True,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.v_proj.bias",
        )

    def init_o(self):
        o_split_n_embed = self.head_dim * self.network_config_["num_attention_heads"] // self.world_size_
        self.o_proj = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            self.data_type_,
            o_split_n_embed,
            bias_name=f"model.layers.{self.layer_num_}.self_attn.o_proj.bias",
        )

    def init_ffn(self):
        n_embed = self.network_config_["hidden_size"]
        intermediate_size = n_embed * 4
        split_inter_size = intermediate_size // self.world_size_
        self.ffn_1_weight_ = ROWMMWeight(
            f"model.layers.{self.layer_num_}.mlp.c_fc.weight",
            self.data_type_,
            split_inter_size,
            bias_name=f"model.layers.{self.layer_num_}.mlp.c_fc.bias",
        )
        self.ffn_2_weight_ = ROWMMWeight(
            f"model.layers.{self.layer_num_}.mlp.c_proj.weight",
            self.data_type_,
            split_inter_size,
            bias_name=f"model.layers.{self.layer_num_}.mlp.c_proj.bias",
        )
