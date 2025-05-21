import torch
from dataclasses import dataclass, field


@dataclass
class ModelInput:
    batch_size: int
    total_token_num: int
    max_len_in_batch: int
    input_ids: torch.Tensor
    mem_indexes: torch.Tensor
    b_req_idx: torch.Tensor
    b_seq_len: torch.Tensor
    is_prefill: bool = False
    b_ready_cache_len: torch.Tensor = None
    multimodal_params: list = field(default_factory=list)
    hidden_states: torch.Tensor = None


@dataclass
class ModelOutput:
    logits: torch.tensor
    hidden_states: torch.tensor
