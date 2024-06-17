import os
import json
import torch
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek2.layer_weights.transformer_layer_weight import Deepseek2TransformerLayerWeight
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.common.basemodel.layer_weights.hf_load_utils import load_hf_weights

from lightllm.common.basemodel import LlamaTpPartModel
from lightllm.common.mem_utils import select_mem_manager_class
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)

class Deepseek2TpPartModle(LlamaTpPartModel):
    # weight class
    transformer_weight_class = Deepseek2TransformerLayerWeight

    # infer class
    transformer_layer_infer_class = Deepseek2TransformerLayerInfer

    # infer state class
    infer_state_class = Deepseek2InferStateInfo

    def __init__(self, kvargs):
        super().__init__(kvargs)
        return
    
    def _init_config(self):
        super()._init_config()
        return
    
    def _verify_params(self):
        return super()._verify_params()
    

    def _init_mem_manager(self):
        self.mem_manager = select_mem_manager_class(self.mode, "deepseek2")(self.max_total_token_num, 
                                                     dtype=self.data_type,
                                                     head_num=self.config["num_key_value_heads"] // self.world_size_,
                                                     key_head_dim=self.config["qk_nope_head_dim"] + self.config["qk_rope_head_dim"],
                                                     value_head_dim=self.config["qk_nope_head_dim"],
                                                     layer_num=self.config["num_hidden_layers"])
        return
    
    def _init_custom(self):
        return
    
    def _init__weights(self):
        self.pre_post_weight = self.pre_and_post_weight_class(self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode)
        self.trans_layers_weight = [
            self.transformer_weight_class(i, self.tp_rank_, self.world_size_, self.data_type, network_config=self.config, mode=self.mode)
            for i in range(self.config["n_layer"])
        ]
        load_hf_weights(
            self.data_type,
            weight_dir=self.weight_dir_,
            pre_post_layer=self.pre_post_weight,
            transformer_layer_list=self.trans_layers_weight,
            weight_dict=self.weight_dict
        )
        self.pre_post_weight.verify_load()
        [weight.verify_load() for weight in self.trans_layers_weight]    
        return