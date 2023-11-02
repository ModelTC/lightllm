import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np

from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer

class Llama2TransformerLayerInfer(LlamaTransformerLayerInfer):

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        key_value_head_num_ = network_config["num_key_value_heads"]
        assert key_value_head_num_ % self.world_size_ == 0
        self.tp_k_head_num_ = key_value_head_num_ // self.world_size_
        self.tp_v_head_num_ = key_value_head_num_ // self.world_size_
        return