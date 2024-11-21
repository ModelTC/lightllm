import torch
from .base_weight import BaseWeightTpl
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


class MMWeightTpl(BaseWeightTpl):
    def __init__(self, data_type, split_n_embed):
        super().__init__()
        self.data_type_ = data_type
        self.split_n_embed = split_n_embed
        self.quant_method = None
        self.weight = None
        self.bias = None

    def set_quant_method(self, quant_method):
        self.quant_method = quant_method

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

    def _post_load_weights(self):
        if self.quant_method is not None:
            self.weight = self.quant_method.quantize(self.weight.cuda(self.tp_rank_))
            return
        self.weight = self.weight.transpose(0, 1).cuda(self.tp_rank_)


class MMWeight(MMWeightTpl):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(data_type, split_n_embed)
        self.weight_name = weight_name
        self.bias_name = bias_name

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.bias_name is not None:
            load_ok = load_ok and self.bias is not None
        return load_ok


class ROWMMWeight(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights):
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)
        weight = None
        if self.weight_name in weights:
            weight = weights[self.weight_name].to(self.data_type_)
            self.weight = weight[start:end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)[start:end]
            self.bias = bias.cuda(self.tp_rank_)
        if weight is None:
            return
        self._post_load_weights()
        return


class COLMMWeight(MMWeight):
    def __init__(self, weight_name, data_type, split_n_embed, bias_name=None):
        super().__init__(weight_name, data_type, split_n_embed, bias_name)

    def load_hf_weights(self, weights):
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)
        weight = None
        if self.weight_name in weights:
            weight = weights[self.weight_name].to(self.data_type_)
            self.weight = weight[:, start:end]
        if self.bias_name in weights:
            bias = weights[self.bias_name].to(self.data_type_)
            self.bias = (bias / self.world_size_).cuda(self.tp_rank_)
        if weight is None:
            return
        self._post_load_weights()
        return


class MultiMMWeight(MMWeightTpl):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=None):
        super().__init__(data_type, split_n_embed)
        self.weight_names = weight_names
        self.bias_names = bias_names
        self.weights = [None] * len(self.weight_names)
        self.biases = [None] * len(self.bias_names)
        self.has_bias = all(b is not None for b in self.bias_names)

    def verify_load(self):
        load_ok = True
        # Verify weight. The weight must be not None.
        load_ok = load_ok and self.weight is not None
        # Verify bias. If bias_name is set, it must be not None.
        if self.has_bias:
            load_ok = load_ok and self.bias is not None
        return load_ok


class MultiROWMMWeight(MultiMMWeight):
    def __init__(self, weight_names, data_type, split_n_embed, bias_names=None):
        super().__init__(weight_names, data_type, split_n_embed, bias_names)

    def _fuse(self):
        if self.weight is None and all(w is not None for w in self.weights):
            self.weight = torch.cat(self.weights, dim=0)
            self._post_load_weights()
        if self.has_bias:
            if self.bias is None and all(b is not None for b in self.biases):
                self.bias = torch.cat(self.biases, dim=0).cuda(self.tp_rank_)
        return self

    def load_hf_weights(self, weights):
        start = self.split_n_embed * self.tp_rank_
        end = self.split_n_embed * (self.tp_rank_ + 1)
        weight = None
        for i in range(len(self.weight_names)):
            if self.weight_names[i] in weights:
                weight = weights[self.weight_names[i]].to(self.data_type_)
                self.weights[i] = weight[start:end]
            if self.has_bias and self.bias_names[i] in weights:
                bias = weights[self.bias_names[i]].to(self.data_type_)
                self.biases[i] = bias[start:end]
        self._fuse()
        return


class CustomMMWeight(ROWMMWeight):
    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
        wait_fuse=False,
        disable_tp=False,
        custom_load=None,
        custom_fuse=None,
    ):
        super().__init__(weight_name, data_type, split_n_embed, bias_name, wait_fuse=wait_fuse, disable_tp=disable_tp)
        self.custom_load = custom_load
        self.custom_fuse = custom_fuse

    def fuse(self, B, op=None):
        if self.custom_fuse is None:
            super().fuse(B, op)
        else:
            weight = self.custom_fuse(self, B)
            self.post_load_weights(weight)

    def load_hf_weights(self, weights):
        if self.custom_load is None:
            super().load_hf_weights(weights)
        else:
            weight = None
            if self.weight_name in weights:
                weight = self.custom_load(self, self.pre_load_weights(weights[self.weight_name]))
            if weight is None:
                return
            if self.wait_fuse:
                self.weight = weight
                return
            self.post_load_weights(weight)
        return


class CustomBMMWeight(CustomMMWeight):
    def __init__(
        self,
        weight_name,
        data_type,
        split_n_embed,
        bias_name=None,
        wait_fuse=False,
        disable_tp=False,
        custom_load=None,
        custom_fuse=None,
    ):
        super().__init__(
            weight_name,
            data_type,
            split_n_embed,
            bias_name,
            wait_fuse=wait_fuse,
            disable_tp=disable_tp,
            custom_load=custom_load,
            custom_fuse=custom_fuse,
        )

    def set_quant_method(self, quant_method):
        return
        raise NotImplementedError("BMM does not currently support quantification")

    def bmm(self, input_tensor, out=None, use_custom_tensor_mananger=True):
        if self.quant_method is not None:
            return self.quant_method.apply(input_tensor, self.weight, self.bias, out)
        if out is None:
            shape = (input_tensor.shape[0], input_tensor.shape[1], self.weight.shape[2])
            dtype = input_tensor.dtype
            device = input_tensor.device
            if use_custom_tensor_mananger:
                out = g_cache_manager.alloc_tensor(shape, dtype, device=device, is_graph_out=False)
            else:
                out = torch.empty(shape, dtype=dtype, device=device)
        if self.bias is None:
            return torch.bmm(input_tensor, self.weight, out=out)
        return torch.addbmm(self.bias, input_tensor, self.weight, out=out)

    def post_load_weights(self, weight):
        if self.quant_method is not None:
            self.weight = self.quant_method.quantize(weight.cuda(self.tp_rank_))
            return
        self.weight = weight.cuda(self.tp_rank_)
