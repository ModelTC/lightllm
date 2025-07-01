import torch
from typing import final
from lightllm.models.registry import ModelRegistry
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.utils.log_utils import init_logger
from lightllm.distributed.communication_op import dist_group_manager


logger = init_logger(__name__)


@ModelRegistry("qwen3_moe")
class Qwen3MOEModel(Qwen3TpPartModel):
    # weight class
    transformer_weight_class = Qwen3MOETransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Qwen3MOETransformerLayerInfer

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return

    def _init_custom(self):
        super()._init_custom()
        dist_group_manager.new_deepep_group(256, self.config["hidden_size"])
