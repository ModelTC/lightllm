import torch
from .base_weight import BaseWeightTpl


class NormWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)


class GEMMANormWeight(NormWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)

    def load_hf_weights(self, weights):
        if self.weight_name in weights:
            self.weight = weights[self.weight_name].to(self.data_type_).cuda(self.tp_rank_) + 1
