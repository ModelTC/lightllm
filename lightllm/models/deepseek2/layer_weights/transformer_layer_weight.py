import torch
import math
import numpy as np
from lightllm.common.basemodel import TransformerLayerWeight
from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight, COLMMWeight, NormWeight, CustomMMWeight, FusedMoeWeight
from functools import partial


def fuse_q_kb(self, A, B):
    q_weight_ = A.weight.transpose(0, 1).contiguous().cpu()
    k_b_proj_ = B.weight.contiguous().cpu()
    q_nope_proj_, q_rope_proj_ = torch.split(
        q_weight_.view(-1, self.tp_q_head_num_, self.qk_nope_head_dim + self.qk_rope_head_dim),
        [self.qk_nope_head_dim, self.qk_rope_head_dim],
        dim=-1,
    )
    self.q_rope_proj_ = self._cuda(q_rope_proj_.reshape(-1, self.qk_rope_head_dim * self.tp_q_head_num_))
    q_nope_proj_ = q_nope_proj_.unsqueeze(2).to(torch.float64)

    k_nope_proj_ = k_b_proj_.unsqueeze(0)
    k_nope_proj_ = k_nope_proj_.to(torch.float64)

    return torch.matmul(q_nope_proj_, k_nope_proj_).view(-1, self.tp_q_head_num_ * self.kv_lora_rank).transpose(0, 1)

def fuse_vb_o(self, A, B):
    v_b_proj_ = A.weight
    o_weight_ = B.weight
    return torch.matmul(v_b_proj_.to(torch.float64), o_weight_.to(torch.float64)).view(-1, self.network_config_["hidden_size"]).transpose(0, 1)

def load_kb(self, A):
    kv_b_proj_ = A.weight
    k_b_proj_ = kv_b_proj_.view(self.num_attention_heads, self.qk_nope_head_dim * 2, self.kv_lora_rank)[
        :, : self.qk_nope_head_dim, :
    ]
    return k_b_proj_[self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :].contiguous().to(self.data_type_).cpu()

def load_vb(self, A):
    kv_b_proj_ = A.weight
    v_b_proj_ = kv_b_proj_.T.view(
        self.kv_lora_rank,
        self.num_attention_heads,
        self.qk_nope_head_dim * 2,
    )[:, :, self.qk_nope_head_dim :]
    return v_b_proj_.transpose(0, 1)[
        self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :
    ].contiguous().to(self.data_type_).cpu()

def load_o(self, A):
    o_weight_ = A.weight
    o_weight_ = o_weight_.T.view(self.num_attention_heads, self.qk_nope_head_dim, -1)
    return o_weight_[self.tp_q_head_num_ * self.tp_rank_ : self.tp_q_head_num_ * (self.tp_rank_ + 1), :, :].contiguous().to(self.data_type_).cpu()

class Deepseek2TransformerLayerWeight(TransformerLayerWeight):
    def __init__(self, layer_num, tp_rank, world_size, data_type, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, data_type, network_config, mode)
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
        if self.q_lora_rank is None:
            self.fuse_pairs = {"q_weight_&k_b_proj_": "fuse_qk_weight_"}
        else:
            self.fuse_pairs = {"q_b_proj_&k_b_proj_": "fuse_qk_weight_"}
        self.fuse_pairs["v_b_proj_&o_weight_"] = "fuse_vo_weight_"

        self.init_qkvo()
        if self.is_moe:
            self.init_moe()
        else:
            self.init_ffn()
        self.init_norm()
        return

    def init_qkvo(self):
        q_split_n_embed_with_rope = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.num_attention_heads // self.world_size_
        )
        if self.q_lora_rank is None:
            self.q_weight_ = CustomMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                wait_fuse=True,
                custom_fuse=partial(fuse_q_kb, self),
            )
        else:
            self.q_a_proj_ = ROWMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_a_proj.weight", self.data_type_, self.q_lora_rank, disable_tp=True
            )
            self.q_b_proj = CustomMMWeight(
                f"model.layers.{self.layer_num_}.self_attn.q_b_proj.weight",
                self.data_type_,
                q_split_n_embed_with_rope,
                wait_fuse=True,
                custom_fuse=partial(fuse_q_kb, self),
            )
        self.kv_a_proj_with_mqa_ = ROWMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_a_proj_with_mqa.weight",
            self.data_type_,
            self.kv_lora_rank + self.qk_rope_head_dim,
            disable_tp=True,
        )
        self.k_b_proj_ = CustomMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
            self.data_type_,
            None,
            wait_fuse=True,
            custom_load=partial(load_kb, self)
        )
        self.v_b_proj_ = CustomMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.kv_b_proj.weight",
            self.data_type_,
            None,
            wait_fuse=True,
            custom_load=partial(load_vb, self),
            custom_fuse=partial(fuse_vb_o, self)
        )
        self.o_weight_ = CustomMMWeight(
            f"model.layers.{self.layer_num_}.self_attn.o_proj.weight", self.data_type_, None, wait_fuse=True,
            custom_load=partial(load_o, self)
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
            data_type=self.data_type
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

    def _load_ffn_weights(self, weights):
        if self.is_moe:
            split_inter_size = self.network_config_["moe_intermediate_size"] // self.world_size_
            for i_experts in range(self.n_routed_experts):
                expert_up_proj = None
                expert_gate_proj = None
                expert_gate_up_proj = None
                expert_down_proj = None

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.up_proj.weight" in weights:
                    expert_up_proj = weights[f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.up_proj.weight"][
                        split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :
                    ]
                    self.experts_up_proj[i_experts] = expert_up_proj

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.gate_proj.weight" in weights:
                    expert_gate_proj = weights[
                        f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.gate_proj.weight"
                    ][split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1), :]
                    self.experts_gate_proj[i_experts] = expert_gate_proj

                # if expert_gate_proj is not None and expert_up_proj is not None:
                #     expert_gate_up_proj = torch.cat([expert_gate_proj, expert_up_proj], dim=0)
                #     expert_gate_up_proj = self._cuda(expert_gate_up_proj)

                if f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.down_proj.weight" in weights:
                    expert_down_proj = weights[
                        f"model.layers.{self.layer_num_}.mlp.experts.{i_experts}.down_proj.weight"
                    ][:, split_inter_size * self.tp_rank_ : split_inter_size * (self.tp_rank_ + 1)]
                    expert_down_proj = self._cuda(expert_down_proj)
                    self.w2_list[i_experts] = expert_down_proj
            with self.lock:
                if (
                    hasattr(self, "experts_up_proj")
                    and None not in self.experts_up_proj
                    and None not in self.experts_gate_proj
                    and None not in self.w2_list
                ):

                    w1_list = []
                    for i_experts in range(self.n_routed_experts):
                        expert_gate_up_proj = torch.cat(
                            [self.experts_gate_proj[i_experts], self.experts_up_proj[i_experts]], dim=0
                        )
                        expert_gate_up_proj = self._cuda(expert_gate_up_proj)
                        w1_list.append(expert_gate_up_proj)

                    inter_shape, hidden_size = w1_list[0].shape[0], w1_list[0].shape[1]
                    self.w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
                    inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                    self.w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(
                        len(self.w2_list), inter_shape, hidden_size
                    )

                    delattr(self, "w2_list")
                    delattr(self, "experts_up_proj")
                    delattr(self, "experts_gate_proj")
        return
