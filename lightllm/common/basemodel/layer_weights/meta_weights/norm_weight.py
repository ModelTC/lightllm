import torch
from .base_weight import BaseWeight


class NormWeight(BaseWeight):
    def __init__(self, weight_name, data_type, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
