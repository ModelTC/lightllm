import torch
from abc import ABC, abstractmethod
from lightllm.utils.dist_utils import get_world_size, get_rank


class BaseWeight(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def load_hf_weights(self, weights):
        pass

    @abstractmethod
    def verify_load(self):
        pass


class BaseWeightTpl(BaseWeight):
    def __init__(self):
        self.world_size_ = get_world_size()
        self.tp_rank_ = get_rank()

    def load_hf_weights(self, weights):
        pass

    def verify_load(self):
        pass


# class BaseWeightTpl(BaseWeight):
#     def __init__(self, weight_name, data_type, bias_name):
#         self.weight_name = weight_name
#         self.bias_name = bias_name
#         self.data_type_ = data_type
#         self.world_size_ = get_world_size()
#         self.tp_rank_ = get_rank()
#         self.weight = None
#         self.bias = None

#     def load_hf_weights(self, weights):
#         if self.weight_name in weights:
#             self.weight = weights[self.weight_name].to(self.data_type_).cuda(self.tp_rank_)
#         if self.bias_name in weights:
#             self.bias = weights[self.bias_name].to(self.data_type_).cuda(self.tp_rank_)

#     def verify_load(self):
#         load_ok = True
#         #Verify weight. The weight must be not None.
#         load_ok = load_ok and self.weight is not None
#         #Verify bias. If bias_name is set, it must be not None.
#         if self.bias_name is not None:
#             load_ok = load_ok and self.bias is not None
#         return load_ok
