# Adapted from vllm/model_executor/layers/fused_moe/layer.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Callable, List, Optional, Tuple


class FusedMoE:
    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
    ):
        from lightllm.models.deepseek2.layer_infer.fused_moe import fused_topk, grouped_topk

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
            )
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(
                hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states, gating_output=router_logits, topk=top_k, renormalize=renormalize
            )

        return topk_weights, topk_ids
