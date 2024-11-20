from lightllm.common.basemodel.layer_weights.meta_weights import NormWeight, ROWMMWeight, COLMMWeight
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Starcoder2TransformerLayerWeight(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _init_config(self):
        self.network_config_["intermediate_size"] = self.network_config_["hidden_size"] * 4
        super()._init_config()

    def _init_weight_names(self):
        super()._init_weight_names()
        self._q_bias_name = f"{self.layer_name}.self_attn.q_proj.bias"
        self._k_bias_name = f"{self.layer_name}.self_attn.k_proj.bias"
        self._v_bias_name = f"{self.layer_name}.self_attn.v_proj.bias"
        self._o_bias_name = f"{self.layer_name}.self_attn.o_proj.bias"

        self._up_weight_name = f"{self.layer_name}.mlp.c_fc.weight"
        self._up_bias_name = f"{self.layer_name}.mlp.c_fc.bias"
        self._down_weight_name = f"{self.layer_name}.mlp.c_proj.weight"
        self._down_bias_name = f"{self.layer_name}.mlp.c_proj.bias"

    def _init_ffn(self):
        split_inter_size = self.n_inter // self.world_size_
        self.up_proj = ROWMMWeight(
            self._up_weight_name, self.data_type_, split_inter_size, bias_name=self._up_bias_name, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            self._down_weight_name, self.data_type_, split_inter_size, bias_name=self._down_bias_name
        )
