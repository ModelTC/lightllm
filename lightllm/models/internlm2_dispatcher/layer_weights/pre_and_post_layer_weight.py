import torch
import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Internlm2DispatcherPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, tp_rank, world_size, data_type, network_config, mode):
        super().__init__(tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        vob_size = self.network_config_["vocab_size"]
        split_indexes = np.linspace(0, vob_size, self.world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        if "model.tok_embeddings.weight" in weights:
            self.wte_weight_ = self._cuda(weights["model.tok_embeddings.weight"][split_start:split_end, :])
        if "output.weight" in weights:
            self.lm_head_weight_ = self._cuda(weights["output.weight"][split_start:split_end, :])
        if "model.norm.weight" in weights:
            self.final_norm_weight_ = self._cuda(weights["model.norm.weight"])

        # load classifiier weight
        if "classifier.0.bias" in weights:
            self.cls0_bias_ = self._cuda(weights["classifier.0.bias"])
        if "classifier.0.weight" in weights:
            self.cls0_weight_ = self._cuda(weights["classifier.0.weight"])
        if "classifier.2.bias" in weights:
            self.cls2_bias_ = self._cuda(weights["classifier.2.bias"])
        if "classifier.2.weight" in weights:
            self.cls2_weight_ = self._cuda(weights["classifier.2.weight"])
        if "classifier.4.bias" in weights:
            self.cls4_bias_ = self._cuda(weights["classifier.4.bias"])
        if "classifier.4.weight" in weights:
            self.cls4_weight_ = self._cuda(weights["classifier.4.weight"])

        return
