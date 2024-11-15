from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    BaseWeight,
    ROWMMWeight,
    COLMMWeight,
    NormWeight,
    TpNormWeight,
)


class CohereTransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        self.use_qk_norm = network_config.get("use_qk_norm", False)
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def init_norm(self, weights):
        q_split_head = self.network_config_["num_attention_heads"] // self.world_size_
        k_split_head = self.network_config_["num_key_value_heads"] // self.world_size_

        self.att_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_)

        if self.use_qk_norm:
            self.q_norm_weight_ = TpNormWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_norm.weight", self.data_type_, q_split_head
            )
            self.k_norm_weight_ = TpNormWeight(
                f"model.layers.{self.layer_num_}.self_attn.k_norm.weight", self.data_type_, k_split_head
            )

        return
