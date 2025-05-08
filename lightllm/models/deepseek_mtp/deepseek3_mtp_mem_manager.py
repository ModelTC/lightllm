import torch
from lightllm.common.deepseek2_mem_manager import Deepseek2MemoryManager
from lightllm.utils.log_utils import init_logger
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt

logger = init_logger(__name__)


class Deepseek3MTPMemoryManager(Deepseek2MemoryManager):
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        self.size = size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        self.dtype = dtype
        # profile the max total token num if the size is None
        self.profile_size(mem_fraction)

        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size

        self.can_use_mem_size = self.size

        rank_in_node = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(
            f"MTP_mem_manger_can_use_token_num_{rank_in_node}"
        )

        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        
        self._init_buffers(
            self.size,
            dtype,
            head_num,
            head_dim,
            layer_num,
        )
        self.HOLD_TOKEN_MEMINDEX = self.size

