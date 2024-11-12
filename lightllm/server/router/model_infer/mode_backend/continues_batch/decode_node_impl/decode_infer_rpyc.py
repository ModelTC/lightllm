import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple, Optional
from rpyc.utils.classic import obtain
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.common.basemodel.infer_lock import acquire_lock_until_ready, release_acquired_lock

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDDecodeInferRpcServer(rpyc.Service):
    def __init__(self, backend: ContinuesBatchBackendForDecodeNode) -> None:
        super().__init__()
        self.backend = backend
        return

    def on_connect(self, conn):
        rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{rank_id}")
        return

    def exposed_alloc_to_frozen_some_tokens(self, group_req_id: int, keys: List[int]) -> Optional[List[int]]:
        group_req_id, keys = list(map(obtain, [group_req_id, keys]))
        acquire_lock_until_ready()
        try:
            alloc_token_indexes = self.backend.model.mem_manager.alloc(len(keys))
            if alloc_token_indexes is None:
                return None
            return alloc_token_indexes.detach().cpu().tolist()
        finally:
            release_acquired_lock()

    def exposed_put_kv_received_to_radix_cache(self, group_req_id: int, keys: List[int], values: List[int]):
        group_req_id, keys, values = list(map(obtain, [group_req_id, keys, values]))
        acquire_lock_until_ready()
        radix_cache = self.backend.radix_cache
        key = torch.tensor(keys, dtype=torch.int64, device="cpu")
        value = torch.tensor(values, dtype=torch.int64, device="cpu")
        prefix_len = radix_cache.insert(key, value)
        self.backend.model.mem_manager.free(value[0:prefix_len].cuda())
        release_acquired_lock()
        return

    def exposed_fail_to_realese_forzen_tokens(self, group_req_id: int, keys: List[int], values: List[int]):
        group_req_id, keys, values = list(map(obtain, [group_req_id, keys, values]))
        acquire_lock_until_ready()
        value = torch.tensor(values, dtype=torch.int64, device="cpu")
        self.backend.model.mem_manager.free(value.cuda())
        release_acquired_lock()
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to info_queues ok")
        return
