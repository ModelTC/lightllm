import torch
import math
import numpy as np
from lightllm.utils.log_utils import init_logger
from lightllm.models.bloom.layer_weights.transformer_layer_weight import BloomTransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    COLMMWeight,
    NormWeight,
    CustomMMWeight,
    FusedMoeWeight,
    CustomBMMWeight,
)

logger = init_logger(__name__)


class MixtralTransformerLayerWeight(BloomTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(
            layer_num,
            tp_rank,
            world_size,
            data_type,
            network_config,
            mode,
            quant_cfg=quant_cfg,
            layer_prefix="model.layers",
        )

        self._init_moe()
        return

    def _init_config(self):
        super()._init_config()
        self.n_routed_experts = self.network_config_["num_local_experts"]

    def _init_weight_names(self):
        super()._init_weight_names()
        self.moe_gate_weight_name = f"{self.layer_name}.mlp.gate.weight"
        self.moe_gate_bias_name = None

    def _init_ffn(self, weights):
        pass

    def _init_moe(self, weights):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        self.moe_gate = ROWMMWeight(
            self.moe_gate_weight_name, self.data_type_, 0, bias_name=self.moe_gate_bias_name, disable_tp=True
        )

        self.experts = FusedMoeWeight(
            gate_proj_name="w1",
            down_proj_name="w2",
            up_proj_name="w3",
            weight_prefix=f"{self.layer_name}.block_sparse_moe.experts",
            n_routed_experts=self.n_routed_experts,
            split_inter_size=split_inter_size,
            data_type=self.data_type_,
        )
