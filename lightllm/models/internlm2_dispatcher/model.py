import os
import json
import torch
from lightllm.models.internlm2_dispatcher.layer_weights.pre_and_post_layer_weight import (
    Internlm2DispatcherPreAndPostLayerWeight,
)
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.internlm2_dispatcher.layer_infer.post_layer_infer import Internlm2DispatcherPostLayerInfer


class Internlm2DispatcherTpPartModel(Internlm2TpPartModel):
    # weight class
    pre_and_post_weight_class = Internlm2DispatcherPreAndPostLayerWeight

    post_layer_infer_class = Internlm2DispatcherPostLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
