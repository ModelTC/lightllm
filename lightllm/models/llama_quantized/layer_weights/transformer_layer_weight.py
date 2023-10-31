import math

import numpy as np
import torch
from functools import partial

from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4


class LlamaTransformerLayerWeightQuantized(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], group_size=128):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        quantize_func_dict = {
            'int8weight': quantize_int8,
            'int4weight': partial(quantize_int4, group_size=group_size)
        }
        self.quantize_weight = None
        self.mode = mode
        self.group_size = group_size
        for item in mode:
            if item in quantize_func_dict:
                self.quantize_weight = quantize_func_dict[item]

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
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
            q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"][split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            q_weight_ = q_weight_.transpose(0, 1)
            q_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.scales"]
            if "int8weight" in self.mode:
                q_scales_ = q_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            else:
                q_scales_ = q_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
                q_scales_ = q_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.self_attn.q_proj.zeros" in weights:  # only int4weight
                q_zeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.zeros"][split_n_embed * self.tp_rank_ // 8: split_n_embed * (self.tp_rank_ + 1) // 8, :]
                q_zeros_ = q_zeros_.transpose(0, 1)
            else:
                q_zeros_ = None
        if f"model.layers.{self.layer_num_}.self_attn.k_proj.weight" in weights:
            k_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.weight"][split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            k_weight_ = k_weight_.transpose(0, 1)
            k_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.scales"]
            if "int8weight" in self.mode:
                k_scales_ = k_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            else:
                k_scales_ = k_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
                k_scales_ = k_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.self_attn.k_proj.zeros" in weights:  # only int4weight
                k_zeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.k_proj.zeros"][split_n_embed * self.tp_rank_ // 8: split_n_embed * (self.tp_rank_ + 1) // 8, :]
                k_zeros_ = k_zeros_.transpose(0, 1)
            else:
                k_zeros_ = None
        if f"model.layers.{self.layer_num_}.self_attn.v_proj.weight" in weights:
            v_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.weight"][split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
            v_weight_ = v_weight_.transpose(0, 1)
            v_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.scales"]
            if "int8weight" in self.mode:
                v_scales_ = v_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            else:
                v_scales_ = v_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1), :]
                v_scales_ = v_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.self_attn.v_proj.zeros" in weights:  # only int4weight
                v_zeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.v_proj.zeros"][split_n_embed * self.tp_rank_ // 8: split_n_embed * (self.tp_rank_ + 1) // 8, :]
                v_zeros_ = v_zeros_.transpose(0, 1)
            else:
                v_zeros_ = None

            self.qkv_fused_weight = torch.cat((q_weight_, k_weight_, v_weight_), 1)
            if "int8weight" in self.mode:
                self.qkv_fused_weight_scale = torch.cat((q_scales_, k_scales_, v_scales_), 0)
            else:
                self.qkv_fused_weight_scale = torch.cat((q_scales_, k_scales_, v_scales_), 1)
                self.qkv_fused_weight_zp = torch.cat((q_zeros_, k_zeros_, v_zeros_), 1)

            self.qkv_fused_weight = self.qkv_fused_weight.cuda()
            self.qkv_fused_weight_scale = self.qkv_fused_weight_scale.cuda()
            self.qkv_fused_weight_zp = self.qkv_fused_weight_zp.cuda() if "int4weight" in self.mode else None

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            if "int8weight" in self.mode:
                o_weight_ = o_weight_[:, split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
            else:
                o_weight_ = o_weight_[:, split_n_embed * self.tp_rank_ // 8: split_n_embed * (self.tp_rank_ + 1) // 8]
            self.o_weight_ = o_weight_.transpose(0, 1)
            o_scales_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.scales"]
            if "int8weight" in self.mode:
                o_scales_ = o_scales_[split_n_embed * self.tp_rank_: split_n_embed * (self.tp_rank_ + 1)]
                self.o_weight_scale_ = o_scales_
            else:
                o_scales_ = o_scales_[:, split_n_embed * self.tp_rank_ // self.group_size: split_n_embed * (self.tp_rank_ + 1) // self.group_size]
                self.o_weight_scale_ = o_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.self_attn.o_proj.zeros" in weights:  # only int4weight
                o_zeros_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.zeros"]
                o_zeros_ = o_zeros_[:, split_n_embed * self.tp_rank_ // self.group_size: split_n_embed * (self.tp_rank_ + 1) // self.group_size]
                self.o_weight_zp_ = o_zeros_.transpose(0, 1)
            else:
                self.o_weight_zp_ = None
            self.o_weight_ = self.o_weight_.cuda()
            self.o_weight_scale_ = self.o_weight_scale_.cuda()
            self.o_weight_zp_ = self.o_weight_zp_.cuda() if "int4weight" in self.mode else None

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"])

        inter_size = self.network_config_['intermediate_size']
        split_inter_size = inter_size // self.world_size_

        if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
            gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            gate_proj = gate_proj.transpose(0, 1)
            gate_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.scales"]
            if "int8weight" in self.mode:
                gate_scales_ = gate_scales_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            else:
                gate_scales_ = gate_scales_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
                gate_scales_ = gate_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.mlp.gate_proj.zeros" in weights:  # only int4weight
                gate_zeros_ = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.zeros"][split_inter_size * self.tp_rank_ // 8: split_inter_size * (self.tp_rank_ + 1) // 8, :]
                gate_zeros_ = gate_zeros_.transpose(0, 1)
            else:
                gate_zeros_ = None
        if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
            up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
            up_proj = up_proj.transpose(0, 1)
            up_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.scales"]
            if "int8weight" in self.mode:
                up_scales_ = up_scales_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            else:
                up_scales_ = up_scales_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1), :]
                up_scales_ = up_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.mlp.up_proj.zeros" in weights:  # only int4weight
                up_zeros_ = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.zeros"][split_inter_size * self.tp_rank_ // 8: split_inter_size * (self.tp_rank_ + 1) // 8, :]
                up_zeros_ = up_zeros_.transpose(0, 1)
            else:
                up_zeros_ = None

            self.gate_up_fused_weight = torch.cat((gate_proj, up_proj), 1)
            if "int8weight" in self.mode:
                self.gate_up_fused_weight_scale = torch.cat((gate_scales_, up_scales_), 0)
            else:
                self.gate_up_fused_weight_scale = torch.cat((gate_scales_, up_scales_), 1)
                self.gate_up_fused_weight_zp = torch.cat((gate_zeros_, up_zeros_), 1)

            self.gate_up_fused_weight = self.gate_up_fused_weight.cuda()
            self.gate_up_fused_weight_scale = self.gate_up_fused_weight_scale.cuda()
            self.gate_up_fused_weight_zp = self.gate_up_fused_weight_zp.cuda() if "int4weight" in self.mode else None

        if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
            down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"]
            if "int8weight" in self.mode:
                down_proj = down_proj[:, split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
            else:
                down_proj = down_proj[:, split_inter_size * self.tp_rank_ // 8: split_inter_size * (self.tp_rank_ + 1) // 8]
            self.down_proj = down_proj.transpose(0, 1)
            down_scales_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.scales"]
            if "int8weight" in self.mode:
                down_scales_ = down_scales_[split_inter_size * self.tp_rank_: split_inter_size * (self.tp_rank_ + 1)]
                self.down_proj_scale = down_scales_
            else:
                down_scales_ = down_scales_[:, split_inter_size * self.tp_rank_ // self.group_size: split_inter_size * (self.tp_rank_ + 1) // self.group_size]
                self.down_proj_scale = down_scales_.transpose(0, 1)
            if f"model.layers.{self.layer_num_}.mlp.down_proj.zeros" in weights:  # only int4weight
                down_zeros_ = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.zeros"]
                down_zeros_ = down_zeros_[:, split_inter_size * self.tp_rank_ // self.group_size: split_inter_size * (self.tp_rank_ + 1) // self.group_size]
                self.down_proj_zp = down_zeros_.transpose(0, 1)
            else:
                self.down_proj_zp = None

            self.down_proj = self.down_proj.cuda()
            self.down_proj_scale = self.down_proj_scale.cuda()
            self.down_proj_zp = self.down_proj_zp.cuda() if "int4weight" in self.mode else None
