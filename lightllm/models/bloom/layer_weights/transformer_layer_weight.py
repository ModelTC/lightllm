import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight


def generate_alibi(n_head, dtype=torch.float16):
    """
    This method is originally the `build_alibi_tensor` function
    in `transformers/models/bloom/modeling_bloom.py`
    of the huggingface/transformers GitHub repository.

    Copyright 2023 ModelTC Team
    Copyright 2022 HuggingFace Inc. team and BigScience workshop

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    """

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(n_head))
    head_alibi = slopes.to(dtype)
    return head_alibi


class BloomTransformerLayerWeight(TransformerLayerWeight):
    def __init__(
        self, layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg=None, layer_prefix="h"
    ):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)

        self.layer_name = f"{layer_prefix}.{self.layer_num_}"

        self._init_name()
        self._init_qkv()
        self._init_o()
        self._init_ffn()
        self._init_norm()
        self.set_quantization()
        return

    def _init_name(self):
        self._q_name = f"{self.layer_name}.self_attention.q_proj"
        self._k_name = f"{self.layer_name}.self_attention.k_proj"
        self._v_name = f"{self.layer_name}.self_attention.v_proj"
        self.o_name = f"{self.layer_name}.self_attention.dense"
        self.up_proj_name = f"{self.layer_name}.mlp.dense_h_to_4h"
        self.down_proj_name = f"{self.layer_name}.mlp.dense_4h_to_h"
        self.att_norm_name = f"{self.layer_name}.input_layernorm"
        self.ffn_norm_name = f"{self.layer_name}.post_attention_layernorm"

    def _split_qkv_weight(self, weights):
        n_embed = self.network_config_["n_embed"]
        head_num = self.network_config_["num_attention_heads"]

        if f"{self.layer_name}.self_attention.query_key_value.weight" in weights:
            att_qkv_dense_weight = weights[f"{self.layer_name}.self_attention.query_key_value.weight"].reshape(
                head_num, 3, -1, n_embed
            )
            weights[f"{self._q_name}.weight"] = att_qkv_dense_weight[:, 0, :, :].reshape(-1, n_embed)
            weights[f"{self._k_name}.weight"] = att_qkv_dense_weight[:, 1, :, :].reshape(-1, n_embed)
            weights[f"{self._v_name}.weight"] = att_qkv_dense_weight[:, 2, :, :].reshape(-1, n_embed)
            del weights[f"{self.layer_name}.self_attention.query_key_value.weight"]

        if f"{self.layer_name}.self_attention.query_key_value.bias" in weights:
            att_qkv_dense_bias = weights[f"h.{self.layer_num_}.self_attention.query_key_value.bias"].reshape(
                head_num, 3, -1
            )
            weights[f"{self._q_name}.bias"] = att_qkv_dense_bias[:, 0, :].reshape(-1)
            weights[f"{self._k_name}.bias"] = att_qkv_dense_bias[:, 1, :].reshape(-1)
            weights[f"{self._v_name}.bias"] = att_qkv_dense_bias[:, 2, :].reshape(-1)
            del weights[f"h.{self.layer_num_}.self_attention.query_key_value.bias"]

    def load_hf_weights(self, weights):
        self._split_qkv_weight(weights)
        super().load_hf_weights(weights)
        return

    def init_static_params(self):
        # 计算生成alibi
        head_num = self.network_config_["num_attention_heads"]
        tp_head_num = head_num // self.world_size_
        tmp_alibi = generate_alibi(head_num, dtype=torch.float32)
        assert head_num % self.world_size_ == 0
        self.tp_alibi = tmp_alibi[self.tp_rank_ * tp_head_num : (self.tp_rank_ + 1) * tp_head_num].contiguous().cuda()
        return

    def _init_qkv(self):
        n_embed = self.network_config_["n_embed"]
        split_n_embed = n_embed // self.world_size_
        self.q_proj = ROWMMWeight(
            f"{self._q_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self._q_name}.bias"
        )
        self.k_proj = ROWMMWeight(
            f"{self._k_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self._k_name}.bias", wait_fuse=True
        )
        self.v_proj = ROWMMWeight(
            f"{self._v_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self._v_name}.bias", wait_fuse=True
        )

    def _init_o(self):
        n_embed = self.network_config_["n_embed"]
        split_n_embed = n_embed // self.world_size_
        self.o_proj = COLMMWeight(
            f"{self.o_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self.o_name}.bias"
        )

    def _init_ffn(self):
        n_embed = self.network_config_["n_embed"] * 4
        split_n_embed = n_embed // self.world_size_
        self.up_proj = ROWMMWeight(
            f"{self.up_proj_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self.up_proj_name}.bias"
        )
        self.down_proj = COLMMWeight(
            f"{self.down_proj_name}.weight", self.data_type_, split_n_embed, bias_name=f"{self.down_proj_name}.bias"
        )

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(
            f"{self.att_norm_name}.weight", self.data_type_, bias_name=f"{self.att_norm_name}.bias"
        )
        self.ffn_norm_weight_ = NormWeight(
            f"{self.ffn_norm_name}.weight", self.data_type_, bias_name=f"{self.ffn_norm_name}.bias"
        )
