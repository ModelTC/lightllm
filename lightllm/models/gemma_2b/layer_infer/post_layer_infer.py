import torch

from lightllm.models.gemma_2b.layer_weights.pre_and_post_layer_weight import Gemma_2bPreAndPostLayerWeight
from lightllm.models.gemma_2b.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.models.llama.layer_infer.post_layer_infer import LlamaPostLayerInfer


class Gemma_2bPostLayerInfer(LlamaPostLayerInfer):
    """ """

    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        return

    def _norm(self, input, infer_state, layer_weight: Gemma_2bPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)