import torch
from .base_weight import BaseWeightTpl


class NormWeight(BaseWeightTpl):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
