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

    def generate_balanced_tensor(self, num_tokens):
        # Evenly distribute num_tokens to num_selected experts out of num_experts.
        # Note that the num_selected experts activated by a token cannot be repeated.
        # Performance is not that important, as it is only activated in special scenarios.
        tensor = torch.zeros((num_tokens, self.num_selected), dtype=torch.int, device="cuda")
        expert_load = torch.zeros(self.num_experts, dtype=torch.int, device="cuda")

        for i in range(num_tokens):
            available_experts = torch.arange(self.num_experts, device="cuda")
            selected = []
            for _ in range(self.num_selected):
                current_load = expert_load[available_experts]
                min_load_indices = torch.where(current_load == current_load.min())[0]
                if len(min_load_indices) > 1:
                    # If there are multiple least-loaded experts, select one randomly
                    chosen_index = torch.randint(0, len(min_load_indices), (1,), device="cuda").item()
                    chosen_expert_index = min_load_indices[chosen_index]
                else:
                    chosen_expert_index = min_load_indices[0]
                chosen_expert = available_experts[chosen_expert_index]
                selected.append(chosen_expert)
                # Remove the selected expert from the list of available experts
                available_experts = torch.cat(
                    [available_experts[:chosen_expert_index], available_experts[chosen_expert_index + 1 :]]
                )
                expert_load[chosen_expert] += 1

            tensor[i] = torch.tensor(selected, dtype=torch.int, device="cuda")

        return tensor

    def get_balance_topk_ids(self, num_tokens):
        if num_tokens in self.balanced_tensors:
            return self.balanced_tensors[num_tokens]

        tensor = self.generate_balanced_tensor(num_tokens)
        self.balanced_tensors[num_tokens] = tensor
        return tensor
