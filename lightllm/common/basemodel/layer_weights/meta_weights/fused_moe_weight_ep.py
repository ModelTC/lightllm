import torch
from typing import Optional, Tuple, List, Dict, Any
from lightllm.utils.dist_utils import get_global_world_size, get_global_rank, get_current_device_id
from .fused_moe_weight_tp import FusedMoeWeightTP
from lightllm.common.fused_moe.grouped_fused_moe_ep import fused_experts_impl
from lightllm.distributed import custom_comm_ops


class FusedMoeWeightEP(FusedMoeWeightTP):
    def __init__(
        self,
        gate_proj_name: str,
        down_proj_name: str,
        up_proj_name: str,
        e_score_correction_bias_name: str,
        weight_prefix: str,
        n_routed_experts: int,
        split_inter_size: int,
        data_type: torch.dtype,
        network_config: Dict[str, Any],
        layer_num: int,
        quant_cfg=None,
    ) -> None:
        super().__init__(
            gate_proj_name,
            down_proj_name,
            up_proj_name,
            e_score_correction_bias_name,
            weight_prefix,
            n_routed_experts,
            split_inter_size,
            data_type,
            network_config,
            layer_num,
            quant_cfg,
        )
        global_world_size = get_global_world_size()
        self.global_rank = get_global_rank()
        self.all_routed_experts = self.n_routed_experts
        assert self.n_routed_experts % global_world_size == 0
        self.n_routed_experts = self.n_routed_experts // global_world_size
        self.experts_up_projs = [None] * self.n_routed_experts
        self.experts_gate_projs = [None] * self.n_routed_experts
        self.experts_up_proj_scales = [None] * self.n_routed_experts
        self.experts_gate_proj_scales = [None] * self.n_routed_experts
        self.e_score_correction_bias = None
        self.w2_list = [None] * self.n_routed_experts
        self.w2_scale_list = [None] * self.n_routed_experts
        self.scoring_func = network_config["scoring_func"]
        self.w1 = [None, None]  # weight, weight_scale
        self.w2 = [None, None]  # weight, weight_scale
        self.tp_rank_ = 0

        # init buffer

    def experts(
        self,
        input_tensor,
        router_logits,
        top_k,
        renormalize,
        use_grouped_topk,
        topk_group,
        num_expert_group,
        prefill,
    ):
        from lightllm.common.fused_moe.topk_select import select_experts

        topk_weights, topk_ids = select_experts(
            hidden_states=input_tensor,
            router_logits=router_logits,
            correction_bias=self.e_score_correction_bias,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            scoring_func=self.scoring_func,
        )
        w1, w1_scale = self.w1
        w2, w2_scale = self.w2
        use_fp8_w8a8 = self.quant_method is not None
        fused_experts_impl(
            hidden_states=input_tensor,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_idx=topk_ids.to(torch.long),
            num_experts=self.all_routed_experts,  # number of all experts
            _buffer=custom_comm_ops.ep_buffer,
            prefill=prefill,
            inplace=True,
            use_fp8_w8a8=use_fp8_w8a8,
            use_fp8_all2all=use_fp8_w8a8,
            use_int8_w8a16=False,  # default to False
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            previous_event=None,  # for overlap
        )
        return

    def _load_hf_weights(self, weights):
        n_expert_ep = self.n_routed_experts

        # tp to ep here
        if self.e_score_correction_bias_name in weights:
            self.e_score_correction_bias = self._cuda(weights[self.e_score_correction_bias_name])

        for i_experts_ep in range(n_expert_ep):
            i_experts = i_experts_ep + n_expert_ep * self.global_rank_
            w1_weight = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.weight"
            w2_weight = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.weight"
            w3_weight = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.weight"
            if w1_weight in weights:
                self.experts_gate_projs[i_experts_ep] = weights[w1_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]
            if w3_weight in weights:
                self.experts_up_projs[i_experts_ep] = weights[w3_weight][
                    self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1), :
                ]

            if w2_weight in weights:
                self.w2_list[i_experts_ep] = weights[w2_weight][
                    :, self.split_inter_size * self.tp_rank_ : self.split_inter_size * (self.tp_rank_ + 1)
                ]
        if self.quant_method is not None:
            self._load_weight_scale(weights)
        self._fuse()

    def _load_weight_scale(self, weights: Dict[str, torch.Tensor]) -> None:
        block_size = 1
        if hasattr(self.quant_method, "block_size"):
            block_size = self.quant_method.block_size
        n_expert_ep = self.n_routed_experts
        for i_experts_ep in range(n_expert_ep):
            i_experts = i_experts_ep + n_expert_ep * self.global_rank
            w1_scale = f"{self.weight_prefix}.{i_experts}.{self.w1_weight_name}.{self.weight_scale_suffix}"
            w2_scale = f"{self.weight_prefix}.{i_experts}.{self.w2_weight_name}.{self.weight_scale_suffix}"
            w3_scale = f"{self.weight_prefix}.{i_experts}.{self.w3_weight_name}.{self.weight_scale_suffix}"
            if w1_scale in weights:
                self.experts_gate_proj_scales[i_experts_ep] = weights[w1_scale][
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                    :,
                ]
            if w3_scale in weights:
                self.experts_up_proj_scales[i_experts_ep] = weights[w3_scale][
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                    :,
                ]

            if w2_scale in weights:
                self.w2_scale_list[i_experts_ep] = weights[w2_scale][
                    :,
                    self.split_inter_size
                    // block_size
                    * self.tp_rank_ : self.split_inter_size
                    // block_size
                    * (self.tp_rank_ + 1),
                ]
