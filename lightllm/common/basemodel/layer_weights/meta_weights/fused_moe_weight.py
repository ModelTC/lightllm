import torch
from .base_weight import BaseWeight
from lightllm.utils.dist_utils import get_world_size, get_rank
import threading
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe import fused_experts


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
        self.tp_rank_ = get_rank()
        self.experts_up_projs = [None] * self.n_routed_experts
        self.experts_gate_projs = [None] * self.n_routed_experts
        self.w2_list = [None] * self.n_routed_experts
        self.quant_method = None
        self.lock = threading.Lock()
    
    def set_quant_method(self, quant_method):
        self.quant_method = quant_method
        if self.quant_method is not None:
            self.quant_method.is_moe = True

    def experts(
            self,
            input_tensor,
            router_logits,
            top_k,
            renormalize,
            use_grouped_topk,
            topk_group,
            num_expert_group
        ):
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group
        )
        if self.quant_method is not None:
            fused_experts(input_tensor,
                         w1=self.w1[0],
                         w2=self.w2[0],
                         topk_weights=topk_weights,
                         topk_ids=topk_ids,
                         inplace=False,
                         use_fp8_w8a8=True,
                         use_int8_w8a16=False,
                         w1_scale=self.w1[1],
                         w2_scale=self.w2[1],
                         a1_scale=None,
                         a2_scale=None)
            return
        fused_experts(hidden_states=input_tensor,
            w1=self.w1,
            w2=self.w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True
        )
        
    def fuse(self):
        with self.lock:
            if (
                hasattr(self, "experts_up_projs")
                and None not in self.experts_up_projs
                and None not in self.experts_gate_projs
                and None not in self.w2_list
            ):
                w1_list = []
                for i_experts in range(self.n_routed_experts):
                    expert_gate_up_proj = torch.cat(
                        [self.experts_gate_projs[i_experts], self.experts_up_projs[i_experts]], dim=0
                    )
                    expert_gate_up_proj = expert_gate_up_proj
                    w1_list.append(expert_gate_up_proj)

                inter_shape, hidden_size = w1_list[0].shape[0], w1_list[0].shape[1]
                self.w1 = torch._utils._flatten_dense_tensors(w1_list).view(len(w1_list), inter_shape, hidden_size)
                inter_shape, hidden_size = self.w2_list[0].shape[0], self.w2_list[0].shape[1]
                self.w2 = torch._utils._flatten_dense_tensors(self.w2_list).view(
                    len(self.w2_list), inter_shape, hidden_size
                )
                if self.quant_method is not None:
                    self.w1 = self.quant_method.quantize(self.w1)
                    self.w2 = self.quant_method.quantize(self.w2)
                else:
                    self.w1 = self._cuda(self.w1)
                    self.w2 = self._cuda(self.w2)
                delattr(self, "w2_list")
                delattr(self, "experts_up_projs")
                delattr(self, "experts_gate_projs")

    def load_hf_weights(self, weights):
        for i_experts in range(self.n_routed_experts):
            w1_weight = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.weight"
            w2_weight = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.weight"
            w3_weight = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.weight"

            if w1_weight in weights:
                self.experts_gate_projs[i_experts] = weights[w1_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]
            if w3_weight in weights:
                self.experts_up_projs[i_experts] = weights[w3_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]

            if w2_weight in weights:
                self.w2_list[i_experts] = weights[w2_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]
        
        self.fuse()

            
    def _cuda(self, cpu_tensor):
        if self.tp_rank_ is None:
            return cpu_tensor.contiguous().to(self.data_type_).cuda()
        else:
            return cpu_tensor.contiguous().to(self.data_type_).cuda(self.tp_rank_)
    
    def verify_load(self):
        return self.w1 is not None and self.w2 is not None
