import os
import json
import torch

from lightllm.models.internlm.layer_infer.transformer_layer_infer import InternlmTransformerLayerInfer
from lightllm.models.internlm.layer_weights.transformer_layer_weight import InternlmTransformerLayerWeight
from lightllm.models.llama.model import LlamaTpPartModel


class InternlmTpPartModel(LlamaTpPartModel):
    # weight class
    transformer_weight_class = InternlmTransformerLayerWeight

    # infer class
    transformer_layer_infer_class = InternlmTransformerLayerInfer

    def __init__(self, tp_rank, world_size, weight_dir, max_total_token_num, load_way="HF", mode=[], weight_dict=None, finetune_config=None):
        super().__init__(tp_rank, world_size, weight_dir, max_total_token_num, load_way, mode, weight_dict, finetune_config)
    