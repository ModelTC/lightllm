import torch
from .base_weight import BaseWeight
from lightllm.utils.dist_utils import get_world_size, get_rank


class FusedMoeWeight(BaseWeight):
    def __init__(self, gate_proj_name, down_proj_name, up_proj_name, weight_prefix, n_routed_experts, split_inter_size, data_type):
        super().__init__()
        self.w1_weight_name = gate_proj_name
        self.w2_weight_name = down_proj_name
        self.w3_weight_name = up_proj_name
        self.weight_prefix = weight_prefix
        self.n_routed_experts = n_routed_experts
        self.split_inter_size = split_inter_size
        self.data_type_ = data_type
        self.experts_up_proj = [None] * self.n_routed_experts
        self.experts_gate_proj = [None] * self.n_routed_experts
        self.w2_list = [None] * self.n_routed_experts

    def fuse(self):
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

    def load_hf_weights(self, weights):
        for i_experts in range(self.n_routed_experts):
            expert_up_proj = None
            expert_gate_proj = None
            expert_gate_up_proj = None
            expert_down_proj = None

            w1_weight = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.weight"
            w2_weight = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.weight"
            w3_weight = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.weight"

            if w1_weight in weights:
                expert_up_proj = weights[w1_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.experts_gate_proj[i_experts] = expert_gate_proj

            if w3_weight in weights:
                expert_gate_proj = weights[w3_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]
                self.experts_up_proj[i_experts] = expert_up_proj

            if w2_weight in weights:
                expert_down_proj = weights[w2_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]
                expert_down_proj = self._cuda(expert_down_proj)
                self.w2_list[i_experts] = expert_down_proj
        
        self.fuse()

            
    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_)