import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain
from .prefill_impl import ContinuesBatchBackendForPrefillNode
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDPrefillInferRpcServer(rpyc.Service):
    def __init__(self, backend: ContinuesBatchBackendForPrefillNode) -> None:
        super().__init__()
        self.backend = backend
        return

    # pd 分离模式会使用的一些接口，用于做一些全局信息管理
    def exposed_remove_req_refs_from_prompt_cache(self, group_req_id: int, keys: List[int], values: List[int]):
        rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{rank_id}")

        group_req_id, keys, values = list(map(obtain, [group_req_id, keys, values]))
        radix_cache = self.backend.radix_cache
        key = torch.tensor(keys, dtype=torch.int64, device="cpu", requires_grad=False)
        share_node, kv_len, value_tensor = radix_cache.match_prefix(key, update_refs=False)
        # to do, can be remove, affect performance
        assert len(values) == len(value_tensor)
        if share_node is not None:
            radix_cache.dec_node_ref_counter(share_node)
            self.backend.model.mem_manager.decrease_refs(value_tensor.cuda())

        logger.info(f"prefill node remove frozen tokens for req id: {group_req_id}")
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to info_queues ok")
        return
