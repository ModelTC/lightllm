import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight, CustomMMWeight, FusedMoeWeight, CustomBMMWeight
from functools import partial


def fuse_q_kb(self, A, B):
    q_weight_ = A.weight.transpose(0, 1).contiguous().cpu()
    k_b_proj_ = B.weight.contiguous().cpu()
    q_nope_proj_, q_rope_proj_ = torch.split(
        q_weight_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )
    q_nope_proj_ = q_nope_proj_.unsqueeze(2).to(torch.float64)

    k_nope_proj_ = k_b_proj_.unsqueeze(0)
    k_nope_proj_ = k_nope_proj_.to(torch.float64)

    return self._cuda(torch.matmul(q_nope_proj_, k_nope_proj_).view(-1, self.tp_q_head_num_ * self.kv_lora_rank).transpose(0, 1))

def fuse_vb_o(self, A, B):
    v_b_proj_ = A.weight
    o_weight_ = B.weight.transpose(0, 1).view(self.tp_q_head_num_, self.qk_nope_head_dim, -1).contiguous().to(self.data_type_).cpu()
    return self._cuda(torch.matmul(v_b_proj_.to(torch.float64), o_weight_.to(torch.float64)).view(-1, self.network_config_["hidden_size"]))

def load_q_rope(self, A, q_weight_):
    q_split_n_embed_with_rope = A.split_n_embed
    q_weight_ = q_weight_[q_split_n_embed_with_rope * self.tp_rank_ : q_split_n_embed_with_rope * (self.tp_rank_ + 1), :]
    q_weight_ = q_weight_.transpose(0, 1).contiguous()
    q_nope_proj_, q_rope_proj_ = torch.split(
        q_weight_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )
    return self._cuda(q_rope_proj_.reshape(-1, self.qk_rope_head_dim * self.tp_q_head_num_).transpose(0, 1))

def load_kb(self, A, kv_b_proj_):
    kv_b_proj_ = kv_b_proj_
    k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)[
        :, : self.qk_nope_head_dim, :
    ]
    k_b_proj_ = k_b_proj_[self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :]
    if A.wait_fuse:
        return k_b_proj_.contiguous().to(self.data_type_).cpu()
    return self._cuda(k_b_proj_)

def load_vb(self, A, kv_b_proj_):
    kv_b_proj_ = kv_b_proj_
    v_b_proj_ = kv_b_proj_.T.view(
        self.kv_lora_rank,
        self.num_attention_heads,
        self.qk_nope_head_dim * 2,
    )[:, :, self.qk_nope_head_dim :].transpose(0, 1)
    v_b_proj_ = v_b_proj_[
        self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :
    ]
    if A.wait_fuse:
        return v_b_proj_.contiguous().to(self.data_type_).cpu()
    return self._cuda(v_b_proj_)

class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[], quant_cfg=None, disable_qk_absorb=False, disable_vo_absorb=False):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode, quant_cfg)
        self.is_moe = (
            self.network_config_["n_routed_experts"] is not None
            and self.layer_num_ >= self.network_config_["first_k_dense_replace"]
            and self.layer_num_ % self.network_config_["moe_layer_freq"] == 0
        )
        self.tp_q_head_num_ = network_config["num_attention_heads"] // self.world_size_
        self.n_routed_experts = self.network_config_["n_routed_experts"]
        self.q_lora_rank = self.network_config_["q_lora_rank"]
        self.qk_nope_head_dim = self.network_config_["qk_nope_head_dim"]
        self.qk_rope_head_dim = self.network_config_["qk_rope_head_dim"]
        self.num_attention_heads = self.network_config_["num_attention_heads"]
        self.kv_lora_rank = self.network_config_["kv_lora_rank"]
        self.disable_qk_absorb = disable_qk_absorb
        self.disable_vo_absorb = disable_vo_absorb
        self.fuse_pairs = {}
        if not self.disable_qk_absorb:
            if self.q_lora_rank is None:
                self.fuse_pairs = {"q_weight_&k_b_proj_": "fuse_qk_weight_"}
            else:
                self.fuse_pairs = {"q_b_proj_&k_b_proj_": "fuse_qk_weight_"}
        if not self.disable_vo_absorb:
            self.fuse_pairs["v_b_proj_&o_weight_"] = "fuse_vo_weight_"
        self.fuse_pairs.update({
            "gate_proj&up_proj": "gate_up_proj",
        })

        self.init_qkvo()
        if self.is_moe:
            self.init_moe()
        else:
            self.init_ffn()
        self.init_norm()
        return

    def init_qkvo(self):
        q_split_n_embed = self.qk_nope_head_dim * self.tp_q_head_num_
        q_split_n_embed_with_rope = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
        )
        if self.q_lora_rank is None:
            self.q_weight_ = CustomMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                wait_fuse=not self.disable_qk_absorb,
                custom_fuse=partial(fuse_q_kb, self),
            )
            rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_proj.weight"
        else:
            self.q_a_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight", self.data_type_, self.q_lora_rank, disable_tp=True
            )
            self.q_b_proj_ = CustomMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                wait_fuse=not self.disable_qk_absorb,
                custom_fuse=partial(fuse_q_kb, self),
            )
            rope_weight_name = f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight"
        self.q_rope_proj_ = CustomMMWeight(
            rope_weight_name,
            self.data_type_,
            q_split_n_embed_with_rope,
            custom_load=partial(load_q_rope, self)
        )
        self.kv_a_proj_with_mqa_ = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            disable_tp=True,
        )
        self.k_b_proj_ = CustomBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
            self.data_type_,
            None,
            wait_fuse=not self.disable_qk_absorb,
            custom_load=partial(load_kb, self)
        )
        self.v_b_proj_ = CustomBMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
            self.data_type_,
            None,
            wait_fuse=not self.disable_vo_absorb,
            custom_load=partial(load_vb, self),
            custom_fuse=partial(fuse_vb_o, self)
        )
        self.o_weight_ = COLMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight", self.data_type_, q_split_n_embed, wait_fuse=not self.disable_vo_absorb,
        )

    def _load_mlp(self, mlp_prefix, split_inter_size):
        self.gate_proj = ROWMMWeight(
            f"{mlp_prefix}.gate_proj.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.up_proj = ROWMMWeight(
            f"{mlp_prefix}.up_proj.weight", self.data_type_, split_inter_size, wait_fuse=True
        )
        self.down_proj = COLMMWeight(
            f"{mlp_prefix}.down_proj.weight", self.data_type_, split_inter_size
        )

    def init_moe(self):
        moe_intermediate_size = self.network_config_["moe_intermediate_size"]
        self.moe_gate = ROWMMWeight(
            f"model.layers.{self.layer_num_}.mlp.gate.weight", self.data_type_, moe_intermediate_size, disable_tp=True
        )
        shared_intermediate_size = (
            moe_intermediate_size * self.network_config_["n_shared_experts"]
        )
        shared_split_inter_size = shared_intermediate_size // self.world_size_
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp.shared_experts", shared_split_inter_size)
        
        self.experts = FusedMoeWeight(
            gate_proj_name="gate_proj",
            down_proj_name="down_proj",
            up_proj_name="up_proj",
            weight_prefix=f"model.layers.{self.layer_num_}.mlp.experts",
            n_routed_experts=self.n_routed_experts,
            split_inter_size=moe_intermediate_size // self.world_size_,
            data_type=self.data_type_
        )

    def init_ffn(self):
        inter_size = self.network_config_["intermediate_size"]
        split_inter_size = inter_size // self.world_size_
        self._load_mlp(f"model.layers.{self.layer_num_}.mlp", split_inter_size)

    def init_norm(self):
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
