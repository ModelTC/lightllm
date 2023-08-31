import math

import numpy as np
import torch

from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.quantize import quantize_int8, quantize_int4


class LlamaTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.q_weight_,
                   self.k_weight_,
                   self.v_weight_,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.up_proj,
                   self.gate_proj,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
    
    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            self.k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.k_weight_ = self._cuda(self.k_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            self.v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                                self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            self.v_weight_ = self._cuda(self.v_weight_.transpose(0, 1))
        
        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                            split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))
        return
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            self.up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                         self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.up_proj = self._cuda(self.up_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            self.gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                             self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            self.gate_proj = self._cuda(self.gate_proj.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj = self._cuda(self.down_proj.transpose(0, 1))


class LlamaTransformerLayerWeightQuantized(LlamaTransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        quantize_func_dict = {
            'int8weight': quantize_int8,
            'int4weight': quantize_int4
        }
        self.quantize_weight = None
        for item in mode:
            if item in quantize_func_dict:
                self.quantize_weight = quantize_func_dict[item]

    def verify_load(self):
        errors = "weights load not ok"
        weights = [self.att_norm_weight_,
                   self.qkv_fused_weight,
                   self.o_weight_,
                   self.ffn_norm_weight_,
                   self.gate_up_fused_weight,
                   self.down_proj
                   ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = n_embed // self.world_size_
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            q_weight_ = q_weight_.transpose(0, 1).to(self.data_type_)
            
            self.qkv_fused_weight = torch.empty(n_embed, split_n_embed * 3, dtype=self.data_type_, device='cpu')
            self.qkv_fused_weight[:, :split_n_embed] = q_weight_
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            k_weight_ = k_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_fused_weight[:, split_n_embed:split_n_embed * 2] = k_weight_
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed *
                                                                                           self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            v_weight_ = v_weight_.transpose(0, 1).to(self.data_type_)
            self.qkv_fused_weight[:, split_n_embed * 2:split_n_embed * 3] = v_weight_
            self.qkv_fused_weight, self.qkv_fused_weight_scale = self.quantize_weight(self.qkv_fused_weight)
            self.qkv_fused_weight = self.qkv_fused_weight.cuda()
            self.qkv_fused_weight_scale = self.qkv_fused_weight_scale.to(self.data_type_).cuda()

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"][:,
                                                                                                split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_, self.o_weight_scale_ = self.quantize_weight(self.o_weight_.transpose(0, 1))
            self.o_weight_ = self.o_weight_.cuda()
            self.o_weight_scale_ = self.o_weight_scale_.to(self.data_type_).cuda()
    
    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])
    
        n_embed = self.network_config_["hidden_size"]
        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size *
                                                                                        self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            gate_proj = gate_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_fused_weight = torch.empty(n_embed, split_inter_size * 2, dtype=self.data_type_, device='cpu')
            self.gate_up_fused_weight[:, :split_inter_size] = gate_proj
        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size *
                                                                                    self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            up_proj = up_proj.transpose(0, 1).to(self.data_type_)
            self.gate_up_fused_weight[:, split_inter_size:split_inter_size * 2] = up_proj
            self.gate_up_fused_weight, self.gate_up_fused_weight_scale = self.quantize_weight(self.gate_up_fused_weight)
            self.gate_up_fused_weight = self.gate_up_fused_weight.cuda()
            self.gate_up_fused_weight_scale = self.gate_up_fused_weight_scale.to(self.data_type_).cuda()

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][:,
                                                                                             split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            self.down_proj, self.down_proj_scale = self.quantize_weight(self.down_proj.transpose(0, 1))
            self.down_proj = self.down_proj.cuda()
            self.down_proj_scale = self.down_proj_scale.to(self.data_type_).cuda()
