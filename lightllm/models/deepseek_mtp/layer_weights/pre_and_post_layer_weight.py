import numpy as np
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight


class Deepseek3MTPPreAndPostLayerWeight(LlamaPreAndPostLayerWeight):
    def __init__(self, data_type, network_config, mode):
        super().__init__(data_type, network_config, mode)
        # 与DeepseekV3模型共享
        self.wte_weight_ = None 
        self.lm_head_weight_ = None
        self.final_norm_weight_ = None
        return
    
    def load_hf_weights(self, weights):
        hidden_size = self.network_config_["hidden_size"]
        split_indexes = np.linspace(0, hidden_size, self.tp_world_size_ + 1, dtype=np.int64)
        split_start = split_indexes[self.tp_rank_]
        split_end = split_indexes[self.tp_rank_ + 1]
        
        if "model.layers.0.eh_proj.weight" in weights:
            self.eh_proj_weight_ = self._cuda(weights["model.layers.0.eh_proj.weight"][split_start:split_end, :])
        if "model.layers.0.enorm.weight" in weights:
            self.enorm_weight_ = self._cuda(weights["model.layers.0.enorm.weight"])
        if "model.layers.0.hnorm.weight" in weights:
            self.hnorm_weight_ = self._cuda(weights["model.layers.0.hnorm.weight"])
        return 
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.eh_proj_weight_, self.enorm_weight_, self.hnorm_weight_]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return