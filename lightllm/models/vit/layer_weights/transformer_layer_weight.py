import torch
import math
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight, MultiROWMMWeight


class ViTTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, gpu_id, world_size, data_type, network_config, mode=[], quant_cfg=None):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        self.gpu_id_ = gpu_id
        return

    def _parse_config(self):
        self.padding_hidden_size = self.network_config_["padding_hidden_size"]
        self.qk_norm = self.network_config_["qk_normalization"]
        self.use_ls = self.network_config_.get("use_ls", False)
        self.qkv_bias = self.network_config_.get("qkv_bias", True)
        self.layer_norm_eps = self.network_config_.get("layer_norm_eps", 1e-6)
        self.norm_type = self.network_config_.get("norm_type", "layer_norm")

    def _init_weight_names(self):
        self._att_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"

        self._q_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q.weight"
        self._k_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k.weight"
        self._v_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.v.weight"

        if self.qkv_bias:
            self._q_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q.bias"
            self._k_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k.bias"
            self._v_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.v.bias"
        else:
            self._q_bias_name = None
            self._k_bias_name = None
            self._v_bias_name = None

        self._o_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight"
        self._o_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias"

        self.fc1_weight_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight"
        self.fc1_bias_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias"
        self.fc2_weight_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight"
        self.fc2_bias_name_ = f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias"

        self._ls1_name = f"vision_model.encoder.layers.{self.layer_num_}.ls1"
        self._ls2_name = f"vision_model.encoder.layers.{self.layer_num_}.ls2"

        self._att_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"
        self._ffn_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight"

        if self.norm_type == "layer_norm":
            self._att_norm_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias"
            self._ffn_norm_bias_name = f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias"
        else:
            self._att_norm_bias_name = None
            self._ffn_norm_bias_name = None

        if self.qk_norm:
            self._q_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight"
            self._k_norm_weight_name = f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight"

    def _init_weight(self):
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()

    def _init_qkv(self):
        n_embed = self.network_config_["hidden_size"]
        qkv_split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
        self.qkv_proj = MultiROWMMWeight(
            [self._q_weight_name, self._k_weight_name, self._v_weight_name],
            self.data_type_,
            qkv_split_n_embed,
            bias_names=[self._q_bias_name, self._k_bias_name, self._v_bias_name],
        )

    def _init_o(self):
        n_embed = self.network_config_["hidden_size"]
        o_split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
        self.o_proj = COLMMWeight(self._o_weight_name, self.data_type_, o_split_n_embed, bias_name=self._o_bias_name)

    def _init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        self.ffn_1_proj_ = ROWMMWeight(
            self.fc1_weight_name_,
            self.data_type_,
            split_inter_size,
            bias_name=self.fc1_bias_name_,
        )
        self.ffn_2_proj_ = COLMMWeight(
            self.fc2_weight_name_, self.data_type_, split_inter_size, bias_name=self.fc2_bias_name_
        )

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(
            self._att_norm_weight_name, self.data_type_, bias_name=self._att_norm_bias_name
        )
        self.ffn_norm_weight_ = NormWeight(
            self._ffn_norm_weight_name, self.data_type_, bias_name=self._ffn_norm_bias_name
        )
        if self.qk_norm:
            self.q_norm_weight_ = NormWeight(
                self._q_norm_weight_name, self.data_type_, bias_name=self._q_norm_bias_name
            )
            self.k_norm_weight_ = NormWeight(
                self._k_norm_weight_name, self.data_type_, bias_name=self._q_norm_bias_name
            )

    def load_hf_weights(self, weights):
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight" in weights:
            n_embed = self.network_config_["hidden_size"]
            split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
            att_qkv_dense_weight = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]
            att_qkv_dense_weight = att_qkv_dense_weight.reshape(3, n_embed, -1)
            q_weight_ = F.pad(att_qkv_dense_weight[0, :, :], (0, 0, 0, self.padding_hidden_size))
            q_weight_ = q_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]
            k_weight_ = F.pad(att_qkv_dense_weight[1, :, :], (0, 0, 0, self.padding_hidden_size))
            k_weight_ = k_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]
            v_weight_ = F.pad(att_qkv_dense_weight[2, :, :], (0, 0, 0, self.padding_hidden_size))
            v_weight_ = v_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]
            del weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]
            weights[self._q_weight_name] = q_weight_
            weights[self._k_weight_name] = k_weight_
            weights[self._v_weight_name] = v_weight_

        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias" in weights:
            n_embed = self.network_config_["hidden_size"]
            split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
            att_qkv_dense_bias = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]
            att_qkv_dense_bias = F.pad(att_qkv_dense_bias, (0, self.padding_hidden_size))
            q_bias_ = att_qkv_dense_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            k_bias_ = att_qkv_dense_bias[
                n_embed + split_n_embed * self.tp_rank_ : n_embed + split_n_embed * (self.tp_rank_ + 1)
            ]
            v_bias_ = att_qkv_dense_bias[
                n_embed * 2 + split_n_embed * self.tp_rank_ : n_embed * 2 + split_n_embed * (self.tp_rank_ + 1)
            ]
            weights[self._q_bias_name] = q_bias_
            weights[self._k_bias_name] = k_bias_
            weights[self._v_bias_name] = v_bias_
            del weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]

        if f"vision_model.encoder.layers.{self.layer_num_}.ls1" in weights:
            ls1 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls1"]
            self.ls1 = self._cuda(ls1)

        if f"vision_model.encoder.layers.{self.layer_num_}.ls2" in weights:
            ls2 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls2"]
            self.ls2 = self._cuda(ls2)
            self.use_ls = True

        return super().load_hf_weights(weights)

    # def _load_qkvo_weights(self, weights):
    #     # input layernorm params
    #     if f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight" in weights:
    #         self.att_norm_weight_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"])
    #     if f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias" in weights:
    #         self.att_norm_bias_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias"])

    #     n_embed = self.network_config_["hidden_size"]
    #     split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight" in weights:
    #         q_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight"]
    #         q_norm_weight_ = F.pad(q_norm_weight_, (0, self.padding_hidden_size))
    #         self.q_norm_weight_ = self._cuda(
    #             q_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
    #         )

    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight" in weights:
    #         k_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight"]
    #         k_norm_weight_ = F.pad(k_norm_weight_, (0, self.padding_hidden_size))
    #         self.k_norm_weight_ = self._cuda(
    #             k_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
    #         )

    #     # q k v weights for llama
    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight" in weights:
    #         att_qkv_dense_weight = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]

    #         att_qkv_dense_weight = att_qkv_dense_weight.reshape(3, n_embed, -1)
    #         # self.qkv_weight_ = self._cuda(att_qkv_dense_weight).t()

    #         q_weight_ = F.pad(att_qkv_dense_weight[0, :, :], (0, 0, 0, self.padding_hidden_size))
    #         self.q_weight_ = q_weight_.reshape(-1, n_embed)[
    #             split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
    #         ]

    #         k_weight_ = F.pad(att_qkv_dense_weight[1, :, :], (0, 0, 0, self.padding_hidden_size))
    #         self.k_weight_ = k_weight_.reshape(-1, n_embed)[
    #             split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
    #         ]

    #         v_weight_ = F.pad(att_qkv_dense_weight[2, :, :], (0, 0, 0, self.padding_hidden_size))
    #         self.v_weight_ = v_weight_.reshape(-1, n_embed)[
    #             split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
    #         ]

    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias" in weights:
    #         att_qkv_dense_bias = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]
    #         # self.qkv_bias_ = self._cuda(att_qkv_dense_bias)
    #         self.q_bias_ = att_qkv_dense_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
    #         self.k_bias_ = att_qkv_dense_bias[
    #             n_embed + split_n_embed * self.tp_rank_ : n_embed + split_n_embed * (self.tp_rank_ + 1)
    #         ]
    #         self.v_bias_ = att_qkv_dense_bias[
    #             n_embed * 2 + split_n_embed * self.tp_rank_ : n_embed * 2 + split_n_embed * (self.tp_rank_ + 1)
    #         ]

    #     self._try_cat_to(["q_weight_", "k_weight_", "v_weight_"], "qkv_weight_", cat_dim=0,
    # handle_func=self._transpose)
    #     self._try_cat_to(["q_bias_", "k_bias_", "v_bias_"], "qkv_bias_", cat_dim=0)
    #     # attention output dense params
    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight" in weights:
    #         o_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.weight"]
    #         o_weight_ = F.pad(o_weight_, (0, self.padding_hidden_size, 0, 0))
    #         o_weight_ = o_weight_[:, split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
    #         # print(o_weight_.shape, o_weight_)
    #         self.o_weight_ = self._cuda(o_weight_).t()
    #     if f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias" in weights:
    #         o_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.proj.bias"]
    #         if self.tp_rank_ == 0:
    #             self.o_bias_ = self._cuda(o_bias_)
    #         else:
    #             self.o_bias_ = self._cuda(torch.zeros_like(o_bias_))

    #     if f"vision_model.encoder.layers.{self.layer_num_}.ls1" in weights:
    #         ls1 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls1"]
    #         self.ls1 = self._cuda(ls1)
    #         self.use_ls = True

    #     # self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

    #     return

    # def _load_ffn_weights(self, weights):
    #     if f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight" in weights:
    #         self.ffn_norm_weight_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight"])

    #     if f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias" in weights:
    #         self.ffn_norm_bias_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias"])

    #     inter_size = self.network_config_["intermediate_size"]
    #     split_inter_size = inter_size // self.world_size_

    #     if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight" in weights:
    #         fc1_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.weight"][
    #             split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
    #         ]
    #         self.fc1_weight_ = self._cuda(fc1_weight_).t()

    #     if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias" in weights:
    #         fc1_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc1.bias"][
    #             split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
    #         ]
    #         self.fc1_bias_ = self._cuda(fc1_bias_)

    #     if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight" in weights:
    #         fc2_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.weight"][
    #             :, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)
    #         ]
    #         self.fc2_weight_ = self._cuda(fc2_weight_).t()

    #     if f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias" in weights:
    #         fc2_bias_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.mlp.fc2.bias"]
    #         if self.tp_rank_ == 0:
    #             self.fc2_bias_ = self._cuda(fc2_bias_)
    #         else:
    #             self.fc2_bias_ = self._cuda(torch.zeros_like(fc2_bias_))

    #     if f"vision_model.encoder.layers.{self.layer_num_}.ls2" in weights:
    #         ls2 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls2"]
    #         self.ls2 = self._cuda(ls2)

    #     return
