# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/distributed/communication_op.py
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


import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from typing import List, Dict, Optional, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.dist_utils import (
    get_current_device_id,
    get_node_world_size,
    get_global_world_size,
    get_dp_world_size,
)

original_all_gather_into_tensor = torch.distributed.all_gather_into_tensor

from contextlib import nullcontext, contextmanager

logger = init_logger(__name__)

try:
    HAS_VLLM = True
    from .custom_all_reduce import CustomAllreduce
except:
    HAS_VLLM = False
    logger.info("vllm or lightllm_kernel is not installed, you can't use custom allreduce")

try:
    HAS_LIGHTLLM_KERNEL = True
    from .custom_all_gather import CustomAllgather
except:
    HAS_LIGHTLLM_KERNEL = False
    logger.info("lightllm_kernel is not installed, you can't use custom allgather")

try:
    import deep_ep

    HAS_DEEPEP = True
except:
    HAS_DEEPEP = False
    logger.info("deep_ep is not installed, you can't use the api of it.")


class CustomProcessGroup:
    def __init__(self, group_num: int = 0, disable_custom_op: bool = False):
        self.group_num = group_num
        self.custom_reduce = None
        self.custom_gather = None
        self.world_size = get_dp_world_size()
        ranks = list(range(self.world_size))
        self.device_group = dist.new_group(ranks, backend="nccl")
        if not disable_custom_op:
            self.init_custom_gather()
            self.init_custom_reduce()

    def init_custom_reduce(self) -> None:
        if not HAS_VLLM or not has_nvlink() or self.world_size not in [2, 4, 6, 8]:
            return
        args = get_env_start_args()
        if args.disable_custom_allreduce:
            return
        ranks = list(range(self.world_size))
        cpu_group = dist.new_group(ranks, backend="gloo")
        self.custom_reduce = CustomAllreduce(cpu_group, torch.cuda.current_device())
        logger.info("Enable VLLM ALLReduce.")

    def init_custom_gather(self) -> None:
        if not HAS_LIGHTLLM_KERNEL or not has_nvlink() or self.world_size not in [2, 4, 6, 8]:
            return

        args = get_env_start_args()
        if not args.enable_custom_allgather:
            return
        ranks = list(range(self.world_size))
        cpu_group = dist.new_group(ranks, backend="gloo")
        self.custom_gather = CustomAllgather(cpu_group, torch.cuda.current_device())
        logger.info("Enable Custom ALLGather.")

    def all_reduce(self, input_: torch.Tensor) -> None:
        if self.custom_reduce is not None and self.custom_reduce.should_custom_ar(input_):
            input_.data = self.custom_reduce.custom_all_reduce(input_)
        else:
            dist.all_reduce(input_, group=self.device_group)
        return

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        if self.custom_gather is not None and self.custom_gather.should_custom_ar(input_):
            self.custom_gather.custom_all_gather(output_, input_)
        else:
            dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)
        return


@contextmanager
def lightllm_capture_graph(group: CustomProcessGroup = None):
    assert group is not None, "dist group should not be None"
    if group.custom_reduce is not None:
        with group.custom_reduce.capture():
            if group.custom_gather is not None:
                with group.custom_gather.capture():
                    yield
            else:
                yield
    else:
        yield


class DistributeGroupManager:
    def __init__(self):
        self.groups = []

    def __len__(self):
        return len(self.groups)

    def new_group(self, disable_custom_op: bool = False) -> CustomProcessGroup:
        group = CustomProcessGroup(group_num=len(self.groups), disable_custom_op=disable_custom_op)
        self.groups.append(group)
        return group

    def get_default_group(self) -> CustomProcessGroup:
        return self.groups[0]

    def get_group(self, group_num: int) -> CustomProcessGroup:
        return self.groups[group_num]

    def new_deepep_group(self, n_routed_experts):
        moe_mode = os.getenv("MOE_MODE", "TP")
        num_max_dispatch_tokens_per_rank = os.getenv("NUM_MAX_DISPATCH_TOKENS_PER_RANK", 128)
        if moe_mode == "TP":
            self.ep_buffer = None
            return
        assert HAS_DEEPEP, "deep_ep is required for expert parallelism"
        global_world_size = get_global_world_size()
        deepep_group = dist.new_group(list(range(global_world_size)))
        test_ll_compatibility, num_rdma_bytes = True, 0
        if test_ll_compatibility:
            self.ll_num_tokens, self.ll_hidden, self.ll_num_experts, _ = num_max_dispatch_tokens_per_rank, 7168, 256, 8
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.ll_num_tokens, self.ll_hidden, global_world_size, self.ll_num_experts
            )
        self.ep_buffer = deep_ep.Buffer(
            deepep_group,
            int(1e9),
            num_rdma_bytes,
            low_latency_mode=test_ll_compatibility,
            num_qps_per_rank=(n_routed_experts // global_world_size if test_ll_compatibility else 1),
        )

    def clear_deepep_buffer(self):
        if hasattr(self, "ep_buffer") and self.ep_buffer is not None:
            self.ep_buffer.clean_low_latency_buffer(self.ll_num_tokens, self.ll_hidden, self.ll_num_experts)


def tensor_parallel_all_reduce(
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
) -> None:
    if isinstance(group, CustomProcessGroup):
        group.all_reduce(input_)
        return
    dist.all_reduce(input_, op, group, async_op)


def tensor_parallel_all_gather_into_tensor(
    output_: torch.Tensor,
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if isinstance(group, CustomProcessGroup):
        group.all_gather_into_tensor(output_, input_)
        return
    dist.all_gather_into_tensor(output_, input_, group, async_op)


def tensor_parallel_all_gather(
    output_: List[torch.Tensor],
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    dist.all_gather(output_, input_, group.device_group, async_op)


dist_group_manager = DistributeGroupManager()
