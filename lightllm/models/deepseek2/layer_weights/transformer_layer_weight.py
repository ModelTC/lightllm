import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        return
    
    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return
    
    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_weight_,
            self.kv_a_proj_with_mqa_,
            self.kv_a_layernorm_,
            self.kv_b_proj_,
            self.o_weight_,
            self.ffn_norm_weight_,
        ]
        if self.is_moe:
            assert len(self.experts) == self.n_routed_experts // self.world_size_, "experts weight load not ok"
            weights.append([
                self.moe_gate,
                self.shared_experts_gate_up_proj,
                self.shared_experts_down_proj
            ])
        else:
            weights.append([
                self.gate_up_proj,
                self.down_proj,
            ])
        
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"model.layers.{self.layer_num_}.input_layernorm.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"model.layers.{self.layer_num_}.input_layernorm.weight"])

        n_embed = self.network_config_["hidden_size"]
        q_split_n_embed = n_embed // self.world_size_
        kv_split_n_embed = (
            n_embed
            // self.network_config_["num_attention_heads"]
            * self.network_config_["num_key_value_heads"]
            // self.world_size_
        )
        # q k v weights for llama
        if f"model.layers.{self.layer_num_}.self_attn.q_proj.weight" in weights:
            self.q_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"]
            self.q_weight_ = self.q_weight_[q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1), :]
            self.q_weight_ = self._cuda(self.q_weight_.transpose(0, 1))

        if f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight" in weights:
            kv_a_proj_with_mqa_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight"]
            print("kv_a_proj_with_mqa shape", kv_a_proj_with_mqa_.shape)
            kv_a_proj_with_mqa_ = kv_a_proj_with_mqa_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.kv_a_proj_with_mqa_ = kv_a_proj_with_mqa_.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight" in weights:
            kv_a_layernorm_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight"]
            kv_a_layernorm_ = kv_a_layernorm_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.kv_a_layernorm_ = kv_a_layernorm_.transpose(0, 1)

        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            kv_b_proj_ = kv_b_proj_[kv_split_n_embed * self.tp_rank_ : kv_split_n_embed * (self.tp_rank_ + 1), :]
            self.kv_b_proj_ = kv_b_proj_.transpose(0, 1)

        # attention output dense params
        if f"model.layers.{self.layer_num_}.self_attn.o_proj.weight" in weights:
            self.o_weight_ = weights[f"model.layers.{self.layer_num_}.self_attn.o_proj.weight"]
            self.o_weight_ = self.o_weight_[:, q_split_n_embed * self.tp_rank_ : q_split_n_embed * (self.tp_rank_ + 1)]
            self.o_weight_ = self._cuda(self.o_weight_.transpose(0, 1))

        return

    def _load_ffn_weights(self, weights):
        if f"model.layers.{self.layer_num_}.post_attention_layernorm.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"model.layers.{self.layer_num_}.post_attention_layernorm.weight"]
            )

        if self.is_moe:
            experts_per_rank = self.n_routed_experts // self.world_size_

            if f"model.layers.{self.layer_num_}.mlp.gate.weight" in weights:
                moe_gate = weights[f"model.layers.{self.layer_num_}.mlp.gate.weight"]
                self.moe_gate = self._cuda(moe_gate.transpose(0, 1))

            if f"model.layers.{self.layer_num_}.shared_experts.up_proj.weight" in weights:
                shared_experts_up_proj = weights[f"model.layers.{self.layer_num_}.shared_experts.up_proj.weight"]
                self.shared_experts_up_proj = shared_experts_up_proj.transpose(0, 1)

            if f"model.layers.{self.layer_num_}.shared_experts.gate_proj.weight" in weights:
                shared_experts_gate_proj = weights[f"model.layers.{self.layer_num_}.shared_experts.gate_proj.weight"]
                self.shared_experts_gate_proj = shared_experts_gate_proj.transpose(0, 1)

            self._try_cat_to(["shared_experts_gate_proj", "shared_experts_up_proj"], "shared_experts_gate_up_proj", cat_dim=1)

            if f"model.layers.{self.layer_num_}.shared_experts.down_proj.weight" in weights:
                self.shared_experts_down_proj = weights[f"model.layers.{self.layer_num_}.shared_experts.down_proj.weight"]
                self.shared_experts_down_proj = self._cuda(self.shared_experts_down_proj.transpose(0, 1))
            
            self.experts = []
            for i_experts in range(experts_per_rank * self.tp_rank_, experts_per_rank * (self.tp_rank_ + 1)):
                self.expert_up_proj = None
                self.expert_gate_proj = None
                self.expert_down_proj = None

                if f"model.layers.{self.layer_num_}.experts.{i_experts}.up_proj.weight" in weights:
                    expert_up_proj = weights[f"model.layers.{self.layer_num_}.experts.{i_experts}.up_proj.weight"]
                    self.expert_up_proj = expert_up_proj.transpose(0, 1)

                if f"model.layers.{self.layer_num_}.experts.{i_experts}.gate_proj.weight" in weights:
                    expert_gate_proj = weights[f"model.layers.{self.layer_num_}.experts.{i_experts}.gate_proj.weight"]
                    self.expert_gate_proj = expert_gate_proj.transpose(0, 1)

                self._try_cat_to(["expert_gate_proj", "expert_up_proj"], "expert_gate_up_proj", cat_dim=1)

                if f"model.layers.{self.layer_num_}.experts.{i_experts}.down_proj.weight" in weights:
                    self.expert_down_proj = weights[f"model.layers.{self.layer_num_}.experts.{i_experts}.down_proj.weight"]
                    self.expert_down_proj = self._cuda(self.expert_down_proj.transpose(0, 1))
                
                if self.expert_gate_up_proj is not None and self.expert_down_proj is not None:
                    self.experts.append({
                        "expert_gate_up_proj": self.expert_gate_up_proj,
                        "expert_down_proj": self.expert_down_proj
                    })

        else:
            inter_size = self.network_config_["intermediate_size"]
            split_inter_size = inter_size // self.world_size_

            if f"model.layers.{self.layer_num_}.mlp.up_proj.weight" in weights:
                up_proj = weights[f"model.layers.{self.layer_num_}.mlp.up_proj.weight"][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.up_proj = up_proj.transpose(0, 1)

            if f"model.layers.{self.layer_num_}.mlp.gate_proj.weight" in weights:
                gate_proj = weights[f"model.layers.{self.layer_num_}.mlp.gate_proj.weight"][
                    split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.gate_proj = gate_proj.transpose(0, 1)

            self._try_cat_to(["gate_proj", "up_proj"], "gate_up_proj", cat_dim=1)

            if f"model.layers.{self.layer_num_}.mlp.down_proj.weight" in weights:
                self.down_proj = weights[f"model.layers.{self.layer_num_}.mlp.down_proj.weight"][
                    :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
                ]
                self.down_proj = self._cuda(self.down_proj.transpose(0, 1))
        return
