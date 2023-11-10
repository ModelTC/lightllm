import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class LlavaPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        super().load_hf_weights(weights)
        if 'model.mm_projector.0.weight' in weights:
            self.mm_projector_0_weight = self._cuda(weights['model.mm_projector.0.weight'])

        if 'model.mm_projector.0.bias' in weights:
            self.mm_projector_0_bias = self._cuda(weights['model.mm_projector.0.bias'])

        if 'model.mm_projector.2.weight' in weights:
            self.mm_projector_2_weight = self._cuda(weights['model.mm_projector.2.weight'])

        if 'model.mm_projector.2.bias' in weights:
            self.mm_projector_2_bias = self._cuda(weights['model.mm_projector.2.bias'])


    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.wte_weight_,
                   self.lm_head_weight_,
                   self.final_norm_weight_,
                   self.mm_projector_0_weight,
                   self.mm_projector_0_bias,
                   self.mm_projector_2_weight,
                   self.mm_projector_2_bias,
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

