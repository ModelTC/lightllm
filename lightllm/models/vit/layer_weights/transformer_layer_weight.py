import torch
import math
import numpy as np
import torch.nn.functional as F
from lightllm.common.basemodel import TransformerLayerWeight


class ViTTransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, gpu_id, world_size, data_type, network_config, mode=[], quant_cfg=None):
        self.padding_hidden_size = network_config["padding_hidden_size"]
        self.qk_norm = network_config["qk_normalization"]
        self.use_ls = network_config.get("use_ls", False)
        self.qkv_bias = network_config.get("qkv_bias", True)
        self.layer_norm_eps = network_config.get("layer_norm_eps", 1e-6)
        self.norm_type = network_config.get("norm_type", "layer_norm")
        self.gpu_id_ = gpu_id
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _cuda(self, cpu_tensor):
        return cpu_tensor.contiguous().to(self.data_type_).cuda(self.gpu_id_)

    def _try_cat_to(self, source_tensor_names, dest_name, cat_dim, handle_func=None):
        if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
            with self.lock:
                if all(hasattr(self, src_name) for src_name in source_tensor_names) and not hasattr(self, dest_name):
                    assert all(
                        not getattr(self, name, None).is_cuda for name in source_tensor_names
                    ), "all not cuda tensor"
                    tensors = [getattr(self, name, None) for name in source_tensor_names]
                    ans = torch.cat(tensors, dim=cat_dim)
                    if handle_func is not None:
                        ans = handle_func(ans)
                    else:
                        ans = self._cuda(ans)
                    setattr(self, dest_name, ans)
                    for name in source_tensor_names:
                        delattr(self, name)
        return

    def load_hf_weights(self, weights):
        self._load_qkvo_weights(weights)
        self._load_ffn_weights(weights)
        return

    def post_load(self):
        # merge ls
        ls1 = self.ls1.to(torch.float64)
        self.o_bias_ = (self.o_bias_.to(torch.float64) * ls1).to(self.data_type_)
        self.o_weight_ = (self.o_weight_.to(torch.float64) * ls1.reshape(1, -1)).to(self.data_type_)

        ls2 = self.ls2.to(torch.float64)
        self.fc2_bias_ = (self.fc2_bias_.to(torch.float64) * ls2).to(self.data_type_)
        self.fc2_weight_ = (self.fc2_weight_.to(torch.float64) * ls2.reshape(1, -1)).to(self.data_type_)
        del self.ls1
        del self.ls2
        torch.cuda.empty_cache()

    def _transpose(self, ans):
        return ans.t().cuda(self.gpu_id_)

    def verify_load(self):
        errors = "weights load not ok"
        if not self.qk_norm:
            self.q_norm_weight_ = torch.ones(1).cuda(self.gpu_id_)
            self.k_norm_weight_ = torch.ones(1).cuda(self.gpu_id_)
        if not self.use_ls:
            self.ls1 = 1.0
            self.ls2 = 1.0

        weights = [
            self.att_norm_weight_,
            self.q_norm_weight_,
            self.k_norm_weight_,
            # self.q_weight_,
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
        self.post_load()
        return

    def _load_qkvo_weights(self, weights):
        # input layernorm params
        if f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight" in weights:
            self.att_norm_weight_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm1.weight"])
        if f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias" in weights:
            self.att_norm_bias_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm1.bias"])

        n_embed = self.network_config_["hidden_size"]
        split_n_embed = (n_embed + self.padding_hidden_size) // self.world_size_
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight" in weights:
            q_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.q_norm.weight"]
            q_norm_weight_ = F.pad(q_norm_weight_, (0, self.padding_hidden_size))
            self.q_norm_weight_ = self._cuda(
                q_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            )

        if f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight" in weights:
            k_norm_weight_ = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.k_norm.weight"]
            k_norm_weight_ = F.pad(k_norm_weight_, (0, self.padding_hidden_size))
            self.k_norm_weight_ = self._cuda(
                k_norm_weight_[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            )

        # q k v weights for llama
        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight" in weights:
            att_qkv_dense_weight = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.weight"]

            att_qkv_dense_weight = att_qkv_dense_weight.reshape(3, n_embed, -1)
            # self.qkv_weight_ = self._cuda(att_qkv_dense_weight).t()

            q_weight_ = F.pad(att_qkv_dense_weight[0, :, :], (0, 0, 0, self.padding_hidden_size))
            self.q_weight_ = q_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]

            k_weight_ = F.pad(att_qkv_dense_weight[1, :, :], (0, 0, 0, self.padding_hidden_size))
            self.k_weight_ = k_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]

            v_weight_ = F.pad(att_qkv_dense_weight[2, :, :], (0, 0, 0, self.padding_hidden_size))
            self.v_weight_ = v_weight_.reshape(-1, n_embed)[
                split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1), :
            ]

        if f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias" in weights:
            att_qkv_dense_bias = weights[f"vision_model.encoder.layers.{self.layer_num_}.attn.qkv.bias"]
            # self.qkv_bias_ = self._cuda(att_qkv_dense_bias)
            self.q_bias_ = att_qkv_dense_bias[split_n_embed * self.tp_rank_ : split_n_embed * (self.tp_rank_ + 1)]
            self.k_bias_ = att_qkv_dense_bias[
                n_embed + split_n_embed * self.tp_rank_ : n_embed + split_n_embed * (self.tp_rank_ + 1)
            ]
            self.v_bias_ = att_qkv_dense_bias[
                n_embed * 2 + split_n_embed * self.tp_rank_ : n_embed * 2 + split_n_embed * (self.tp_rank_ + 1)
            ]

        self._try_cat_to(["q_weight_", "k_weight_", "v_weight_"], "qkv_weight_", cat_dim=0, handle_func=self._transpose)
        self._try_cat_to(["q_bias_", "k_bias_", "v_bias_"], "qkv_bias_", cat_dim=0)
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
            self.use_ls = True

        # self._try_cat_to(["k_weight_", "v_weight_"], "kv_weight_", cat_dim=1)

        return

    def _load_ffn_weights(self, weights):
        if f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight" in weights:
            self.ffn_norm_weight_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm2.weight"])

        if f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias" in weights:
            self.ffn_norm_bias_ = self._cuda(weights[f"vision_model.encoder.layers.{self.layer_num_}.norm2.bias"])

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
            if self.tp_rank_ == 0:
                self.fc2_bias_ = self._cuda(fc2_bias_)
            else:
                self.fc2_bias_ = self._cuda(torch.zeros_like(fc2_bias_))

        if f"vision_model.encoder.layers.{self.layer_num_}.ls2" in weights:
            ls2 = weights[f"vision_model.encoder.layers.{self.layer_num_}.ls2"]
            self.ls2 = self._cuda(ls2)

        return
