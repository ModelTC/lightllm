import torch
import math
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import TransformerLayerWeight


class ViTTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        self.padding_hidden_size = network_config["padding_hidden_size"]
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def verify_load(self):
        errors = "weights load not ok"
        weights = [
            self.att_norm_weight_,
            self.q_norm_weight_,
            self.k_norm_weight_,
            self.q_weight_,
            self.o_weight_,
            self.o_bias_,
            self.ffn_norm_weight_,
            self.fc1_weight_,
            self.fc1_bias_,
            self.fc2_weight_,
            self.fc2_bias_,
            self.ls1,
            self.ls2,
        ]
        for i in range(len(weights)):
            assert weights[i] is not None, "index:" + str(i) + " " + errors
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight" in weights:
            q_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight"]
            q_norm_weight_ =  F.pad(q_norm_weight_, (0, self.padding_hidden_size))
            self.q_norm_weight_ = self._cuda(q_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)])


        if f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight" in weights:
            k_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight"]
            k_norm_weight_ =  F.pad(k_norm_weight_, (0, self.padding_hidden_size))
            self.k_norm_weight_ = self._cuda(k_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)])

        # q k v weights for llama
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight" in weights:
            att_qkv_dense_weight = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]

            head_num = self.network_config_["num_attention_heads"]
            att_qkv_dense_weight = att_qkv_dense_weight.reshape(
                3, n_embed, -1
            )
            q_weight_ = F.pad(att_qkv_dense_weight[0 , :, :], (0, 0, 0, self.padding_hidden_size))
            # print(q_weight_.shape, split_n_embed)
            self.q_weight_ = self._cuda(
                                q_weight_.reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                             ).half().t()
        
            # print("q shape ", self.q_weight_.shape, self.padding_hidden_size)

            k_weight_ = F.pad(att_qkv_dense_weight[1 , :, :], (0, 0, 0, self.padding_hidden_size))
            self.k_weight_ = self._cuda(
                                 k_weight_.reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                              ).half().t()

            v_weight_ = F.pad(att_qkv_dense_weight[2 , :, :], (0, 0, 0, self.padding_hidden_size))
            self.v_weight_ = self._cuda(
                                  v_weight_.reshape(-1, n_embed)[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :]
                              ).half().t()
        # self._try_cat_to(["q_weight_", "k_weight_", "v_weight_"], "qkv_weight_", cat_dim=1)
        # attention output dense params
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight" in weights:
            o_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight"]
            o_weight_ = F.pad(o_weight_, (0, self.padding_hidden_size, 0, 0))
            o_weight_ = o_weight_[:, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            # print(o_weight_.shape, o_weight_)
            self.o_weight_ = self._cuda(o_weight_).t()
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias" in weights:
            o_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias"]
            if self.tp_rank_ == 0:
                self.o_bias_ = self._cuda(o_bias_)
            else:
                self.o_bias_ = self._cuda(torch.zeros_like(o_bias_))
        
        if f"vision_model.encoder.layers.{self.layer_num_}.ls1" in weights:
            ls1 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls1"]
            self.ls1 = self._cuda(ls1)

        # self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        return

    def _load_ffn_weights(self, weights):
        if f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(
                weights[f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight"]
            )

        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_

        if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight" in weights:
            fc1_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
            ]
            self.fc1_weight_ = self._cuda(fc1_weight_).t()


        if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias" in weights:
            fc1_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias"][
                split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.fc1_bias_ = self._cuda(fc1_bias_)


        if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight" in weights:
            fc2_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight"][
                :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
            ]
            self.fc2_weight_ = self._cuda(fc2_weight_).t()

        if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias" in weights:
            fc2_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias"]
            self.fc2_bias_ = self._cuda(fc2_bias_)

        if f"vision_model.encoder.layers.{self.layer_num_}.ls2" in weights:
            ls2 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls2"]
            self.ls2 = self._cuda(ls2)

        return
