import torch
import os

import threading

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def singleton_threadsafe(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        # A key that includes the arguments is needed for parameter-dependent singletons.
        # Using a tuple of args and a frozenset of kwargs items makes it hashable.
        key = (cls, args, frozenset(kwargs.items()))
        with lock:
            if key not in instances:
                instances[key] = cls(*args, **kwargs)
            return instances[key]

    return get_instance


@singleton_threadsafe
class BalancedTensor:
    def __init__(self, num_experts=256, num_selected=8):
        self.balanced_tensors = {}
        self.num_experts = num_experts
        self.num_selected = num_selected

    def generate_balanced_tensor(self, length):
        tensor = torch.empty((length, self.num_selected), dtype=torch.int, device="cuda")
        expert_load = torch.zeros(self.num_experts, dtype=torch.int, device="cuda")

        expert_indices = torch.arange(self.num_experts, device="cuda")

        for i in range(length):
            # To break ties randomly when loads are equal, we can shuffle indices
            # of experts with the same load. A simple way is to shuffle all
            # indices and then sort by load.
            shuffled_indices = expert_indices[torch.randperm(self.num_experts, device="cuda")]
            sorted_shuffled_indices = shuffled_indices[torch.argsort(expert_load[shuffled_indices])]

            # Select the top `num_selected` experts with the lowest load
            selected_experts = sorted_shuffled_indices[: self.num_selected]

            tensor[i] = selected_experts

            # Update loads for the selected experts using an efficient scatter_add
            expert_load.scatter_add_(0, selected_experts, torch.ones_like(selected_experts, dtype=torch.int))

        return tensor

    def get_balance_topk_ids(self, num_tokens):
        if self.balanced_tensors.get(num_tokens) is not None:
            # logger.info(f"find balanced tensor for num_tokens={num_tokens}")
            return self.balanced_tensors[num_tokens]
        else:
            # logger.info(f"generate balanced tensor for num_tokens={num_tokens}")
            tensor = self.generate_balanced_tensor(num_tokens)
            self.balanced_tensors[num_tokens] = tensor
            return tensor
