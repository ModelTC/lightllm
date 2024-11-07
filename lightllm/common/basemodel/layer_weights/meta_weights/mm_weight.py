import abc
import torch
from .base_weight import BaseWeight
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


class MMWeight(BaseWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, bias_name)
        self.quant_method = None
        self.split_n_embed = split_n_embed

    def pre_load_weights(self, weight):
        return weight.to(self.data_type_)

    def post_load_weights(self, weight):
        if self.quant_method is not None:
            self.weight = self.quant_method.quantize(weight.cuda(self.tp_rank_))
            return
        self.weight = weight.transpose(0, 1).cuda(self.tp_rank_)

    def mm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        if self.quant_method is not None:
            return self.quant_method.apply(input_tensor, self.weight, self.bias, out)
        if out is None:
            shape = (input_tensor.shape[0], self.weight.shape[1])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.mm(input_tensor, self.weight, out=out)
        return torch.addmm(self.bias, input_tensor, self.weight, out=out)


class ROWMMWeight(MMWeight):
    def __init__(
        self, weight_name, data_type, split_n_embed, bias_name=None, offset=0, wait_fuse=False, disable_tp=False
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)
        self.wait_fuse = wait_fuse
        self.offset = offset
        self.disable_tp = disable_tp

    def fuse(self, B, op="cat"):
        if op == "cat":
            weight = torch.cat([self.weight, B.weight], dim=0)
            if self.bias is not None:
                self.bias = torch.cat([self.bias, B.bias], dim=0)
        elif op == "absorb":
            Aweight = self.weight.to(torch.float64)
            Bweight = B.weight.to(torch.float64)
            weight = torch.matmul(Aweight, Bweight).to(self.data_type_)
        else:
            pass
        self.post_load_weights(weight)
        return self

    def load_hf_weights(self, weights):
        if self.disable_tp:
            rank_id = 0
        else:
            rank_id = self.tp_rank_
        start = self.offset + self.split_n_embed * rank_id
        end = self.offset + self.split_n_embed * (rank_id + 1)

        weight = None
        if self.weight_name in weights:
            weight = self.pre_load_weights(weights[self.weight_name])
            weight = weight[start:end, :]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)[start:end]
            self.bias = bias.cuda(self.tp_rank_)
        if weight is None:
            return
        if self.wait_fuse:
            self.weight = weight
            return
        self.post_load_weights(weight)
        return


class COLMMWeight(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights):
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)
        weight = None
        if self.weight_name in weights:
            weight = self.pre_load_weights(weights[self.weight_name])
            weight = weight[:, start:end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type)
            self.bias = bias.cuda(self.tp_rank_) / self.world_size_
        if weight is None:
            return
        self.post_load_weights(weight)
        return
