from lightllm.models.qwen2.layer_weights.transformer_layer_weight import Qwen2TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import NormWeight


class StablelmTransformerLayerWeight(Qwen2TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

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
