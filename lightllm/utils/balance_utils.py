import torch
import os

import threading

def singleton_threadsafe(cls):
    instances = {}
    lock = threading.Lock()

    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
    return get_instance

@singleton_threadsafe
class BalancedTensor:
    def __init__(self, num_experts=256, num_selected=8):
        self.balanced_tensors = {}
        self.num_experts = num_experts
        self.num_selected = num_selected

    def generate_balanced_tensor(self, length):
        # 初始化一个 length * 8 的全零张量，放置在 GPU 上
        tensor = torch.zeros((length, self.num_selected), dtype=torch.int, device='cuda')
        # 初始化每个专家的负载计数
        expert_load = torch.zeros(self.num_experts, dtype=torch.int, device='cuda')

        for i in range(length):
            available_experts = torch.arange(self.num_experts, device='cuda')
            selected = []
            for _ in range(self.num_selected):
                # 计算每个可用专家的当前负载
                current_load = expert_load[available_experts]
                # 选择负载最小的专家
                min_load_indices = torch.where(current_load == current_load.min())[0]
                if len(min_load_indices) > 1:
                    # 如果有多个负载最小的专家，随机选择一个
                    chosen_index = torch.randint(0, len(min_load_indices), (1,), device='cuda').item()
                    chosen_expert_index = min_load_indices[chosen_index]
                else:
                    chosen_expert_index = min_load_indices[0]
                chosen_expert = available_experts[chosen_expert_index]
                selected.append(chosen_expert)
                # 从可用专家列表中移除已选择的专家
                available_experts = torch.cat(
                    [available_experts[:chosen_expert_index], available_experts[chosen_expert_index + 1:]])
                # 更新该专家的负载
                expert_load[chosen_expert] += 1
            tensor[i] = torch.tensor(selected, dtype=torch.int, device='cuda')
        return tensor

    def get_balance_topk_ids(self, length):
        if self.balanced_tensors.get(length) is not None:
            #print("find length ", length)
            return self.balanced_tensors[length]
        else:
            #print("generate length ", length)
            tensor = self.generate_balanced_tensor(length)
            self.balanced_tensors[length] = tensor
            return tensor

