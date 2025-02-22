import os
import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import (
    ROWMMWeight,
    ROWMMWeightNoTP,
    MultiROWMMWeight,
    MultiROWMMWeightNoTP,
    COLMMWeight,
    COLMMWeightNoTp,
    MultiCOLMMWeight,
    MultiCOLMMWeightNoTp,
    NormWeight,
    FusedMoeWeight,
    ROWBMMWeight,
    ROWBMMWeightNoTp,
)
from functools import partial

import triton
import triton.language as tl
from triton import Config


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE"]), triton.cdiv(N, meta["BLOCK_SIZE"]))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y.to(torch.bfloat16)


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None):
        self.enable_dp = os.getenv("ENABLE_DP", "0").upper() in ["ON", "TRUE", "1"]
        self.enable_cc_method = not os.getenv("DISABLE_CC_METHOD", "False").upper() in ["ON", "TRUE", "1"]
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        return

    def _parse_config(self):
        super()._parse_config()
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.tp_q_head_num_ = self.network_config_["num_attention_heads"]
        if not self.enable_dp:
            self.tp_q_head_num_ = self.tp_q_head_num_ // self.world_size_
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.qk_nope_head_dim = self.network_config_["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.network_config_["qk_rope_head_dim"]
        self.v_head_dim = self.network_config_["v_head_dim"]
        self.num_attention_heads = self.network_config_["num_attention_heads"]
        self.kv_lora_rank = self.network_config_["kv_lora_rank"]

    def _init_weight_names(self):
        if self.q_lora_rank is None:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        else:
            self.rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight"
        self.e_score_correction_bias_name = f"model.layers.{self.layer_num_}.mlp.gate.e_score_correction_bias"

    def _init_qweight_names(self):
        self.act_scale_suffix = None
        self.weight_scale_suffix = None
        if self.quant_cfg.static_activation:
            self.act_scale_suffix = "input_scale"
        if self.quant_cfg.quantized_weight:
            self.weight_scale_suffix = "weight_scale_inv"

    def _init_weight(self):
        if not self.enable_dp:
            self._init_qkvo()
        else:
            self._init_qkvo_dp()
        if self.is_moe:
            self._init_moe()
        else:
            self._init_ffn()
        self._init_norm()

    def _load_q_rope(self, q_weight_):
        if not self.enable_dp:
            q_split_n_embed_with_rope = (
                (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
            )
            q_weight_ = q_weight_[
                q_split_n_embed_with_rope * self.tp_rank_ : q_split_n_embed_with_rope * (self.tp_rank_ + 1), :
            ]
        q_weight_ = q_weight_.transpose(0, 1).contiguous()
        q_nope_proj_, q_rope_proj_ = torch.split(
            q_weight_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        return q_rope_proj_.reshape(-1, self.qk_rope_head_dim * self.tp_q_head_num_).transpose(0, 1).contiguous()

    def _load_kb(self, kv_b_proj_):
        k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)[
            :, : self.qk_nope_head_dim, :
        ]
        return k_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_kb_scale(self, kv_b_proj_, block_size):
        k_b_proj_scale_ = kv_b_proj_.view(
            self.num_attention_heads, self.qk_nope_head_dim * 2 // block_size, self.kv_lora_rank // block_size
        )[:, : self.qk_nope_head_dim // block_size, :]
        return k_b_proj_scale_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb(self, kv_b_proj_):
        v_b_proj_ = kv_b_proj_.T.view(self.kv_lora_rank, self.num_attention_heads, self.qk_nope_head_dim * 2,)[
            :, :, self.qk_nope_head_dim :
        ].transpose(0, 1)
        return v_b_proj_.contiguous().to(kv_b_proj_.dtype)

    def _load_vb_scale(self, kv_b_proj_scale_, block_size):
        v_b_proj_scale_ = kv_b_proj_scale_.T.view(
            self.kv_lora_rank // block_size,
            self.num_attention_heads,
            self.qk_nope_head_dim * 2 // block_size,
        )[:, :, self.qk_nope_head_dim // block_size :].transpose(0, 1)
        return v_b_proj_scale_.contiguous().to(kv_b_proj_scale_.dtype)

    def load_hf_weights(self, weights):
        if f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight" in weights:
            kv_b_proj_ = weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight"]
            f_kv_b_proj_ = weight_dequant(
                kv_b_proj_.cuda(),
                weights[f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + self.weight_scale_suffix].cuda(),
            )
            weights[f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight"] = self._load_kb(f_kv_b_proj_)
            weights[f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight"] = self._load_vb(f_kv_b_proj_)

        if (
            self.quant_cfg.quantized_weight
            and f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + self.weight_scale_suffix in weights
        ):
            kv_b_proj_scale_ = weights[
                f"model.layers.{self.layer_num_}.self_attn.kv_b_proj." + self.weight_scale_suffix
            ]
            block_size = 1
            if self.quant_cfg is not None:
                hf_quantization_config = self.quant_cfg.hf_quantization_config
                block_size = hf_quantization_config.get("weight_block_size", [128, 128])[0]
            weights[
                f"model.layers.{self.layer_num_}.self_attn.k_b_proj." + self.weight_scale_suffix
            ] = self._load_kb_scale(kv_b_proj_scale_, block_size)
            weights[
                f"model.layers.{self.layer_num_}.self_attn.v_b_proj." + self.weight_scale_suffix
            ] = self._load_vb_scale(kv_b_proj_scale_, block_size)

        return super().load_hf_weights(weights)

    def _set_quantization(self):
        super()._set_quantization()
        # moe_gate of deepseek always keep bf16/fp16.
        if self.is_moe:
            self.moe_gate.quant_method = None

    def _init_qkvo(self):
        q_split_n_embed = self.qk_nope_head_dim * self.tp_q_head_num_
        q_split_n_embed_with_rope = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
        )
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.q_a_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                self.data_type_,
                self.q_lora_rank,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.q_b_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.kv_a_proj_with_mqa_ = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )
        self.k_b_proj_ = ROWBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            # weight_scale_suffix=self.weight_scale_suffix,
            # act_scale_suffix=self.act_scale_suffix,
        )
        self.v_b_proj_ = ROWBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            # weight_scale_suffix=self.weight_scale_suffix,
            # act_scale_suffix=self.act_scale_suffix,
        )
        if self.enable_cc_method:
            self.cc_kv_b_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
                self.data_type_,
                split_n_embed=self.tp_q_head_num_ * (self.qk_nope_head_dim + self.v_head_dim),
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.o_weight_ = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            self.data_type_,
            q_split_n_embed,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _init_qkvo_dp(self):
        q_split_n_embed = self.qk_nope_head_dim * self.tp_q_head_num_
        q_split_n_embed_with_rope = (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads
        if self.q_lora_rank is None:
            self.q_weight_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.q_a_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight",
                self.data_type_,
                self.q_lora_rank,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.q_b_proj_ = ROWMMWeightNoTP(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

        self.kv_a_proj_with_mqa_ = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.k_b_proj_ = ROWBMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.k_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.v_b_proj_ = ROWBMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.v_b_proj.weight",
            self.data_type_,
            split_n_embed=self.tp_q_head_num_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

        self.o_weight_ = COLMMWeightNoTp(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight",
            self.data_type_,
            q_split_n_embed,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _load_mlp(self, mlp_prefix, split_inter_size, no_tp=False):
        if no_tp:
            self.gate_up_proj = MultiROWMMWeightNoTP(
                [f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.down_proj = COLMMWeightNoTp(
                f"{mlp_prefix}.down_proj.weight",
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
        else:
            self.gate_up_proj = MultiROWMMWeight(
                [f"{mlp_prefix}.gate_proj.weight", f"{mlp_prefix}.up_proj.weight"],
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )
            self.down_proj = COLMMWeight(
                f"{mlp_prefix}.down_proj.weight",
                self.data_type_,
                split_inter_size,
                weight_scale_suffix=self.weight_scale_suffix,
                act_scale_suffix=self.act_scale_suffix,
            )

    def _init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeightNoTP(
            f"model.layers.{self.layer_num_}.mlp.gate.weight",
            self.data_type_,
            moe_intermediate_size,
            weight_scale_suffix=None,
            act_scale_suffix=None,
        )
        shared_intermediate_size = moe_intermediate_size * self.network_config_["n_shared_experts"]
        shared_split_inter_size = shared_intermediate_size // self.world_size_
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", shared_split_inter_size)

        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            e_score_correction_bias_name=self.e_score_correction_bias_name,
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            split_inter_size=moe_intermediate_size // self.world_size_,
            data_type=self.data_type_,
            network_config=self.network_config_,
            weight_scale_suffix=self.weight_scale_suffix,
            act_scale_suffix=self.act_scale_suffix,
        )

    def _init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        num_shards = self.world_size_ if not self.enable_dp else 1
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp", inter_size // num_shards, no_tp=self.enable_dp)

    def _init_norm(self):
        self.att_norm_weight_ = NormWeight(f"model.layers.{self.layer_num_}.input_layernorm.weight", self.data_type_)
        self.ffn_norm_weight_ = NormWeight(
            f"model.layers.{self.layer_num_}.post_attention_layernorm.weight", self.data_type_
        )
        self.kv_a_layernorm_ = NormWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_layernorm.weight", self.data_type_
        )
        if self.q_lora_rank is not None:
            self.q_a_layernorm_ = NormWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_a_layernorm.weight", self.data_type_
            )
