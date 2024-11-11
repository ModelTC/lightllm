import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple, Optional
from rpyc.utils.classic import obtain
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class PDDecodeInferRpcServer(rpyc.Service):
    def __init__(self, backend: ContinuesBatchBackendForDecodeNode) -> None:
        super().__init__()
        self.backend = backend
        return

    def exposed_alloc_to_frozen_some_tokens(self, group_req_id: int, keys: List[int]) -> Optional[List[int]]:
        rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{rank_id}")

        group_req_id, keys = list(map(obtain, [group_req_id, keys]))
        alloc_token_indexes = self.backend.model.mem_manager.alloc(len(keys))
        if alloc_token_indexes is None:
            return None
        return alloc_token_indexes.detach().cpu().tolist()

    def exposed_put_kv_received_to_radix_cache(self, group_req_id: int, keys: List[int], values: List[int]):
        rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{rank_id}")

        group_req_id, keys, values = list(map(obtain, [group_req_id, keys, values]))
        radix_cache = self.backend.radix_cache
        key = torch.tensor(keys, dtype=torch.int64, device="cpu")
        value = torch.tensor(values, dtype=torch.int64, device="cpu")
        prefix_len = radix_cache.insert(key, value)
        self.backend.model.mem_manager.free(value[0:prefix_len].cuda())
        # 当前的处理，将 token 放入到 radix cache 后并没有进行锁定，其可能会因为缺少显存进行移除叶节点的操作
        # 后续进行相关的优化
        return
        # share_node, kv_len, value_tensor = radix_cache.match_prefix(key, update_refs=True)
        # assert len(keys) == len(value_tensor)

        # need_tokens = len(key) - kv_len
        # alloc_token_indexes = self.backend.model.mem_manager.alloc(need_tokens)
        # if alloc_token_indexes is None:
        #     if share_node is not None:
        #         radix_cache.dec_node_ref_counter(share_node)
        #     return

        # return alloc_token_indexes.detach().cpu().tolist()

    def exposed_fail_to_realese_forzen_tokens(self, group_req_id: int, keys: List[int], values: List[int]):
        rank_id = dist.get_rank()
        torch.cuda.set_device(f"cuda:{rank_id}")

        group_req_id, keys, values = list(map(obtain, [group_req_id, keys, values]))
        value = torch.tensor(values, dtype=torch.int64, device="cpu")
        self.backend.model.mem_manager.free(value.cuda())
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to info_queues ok")
        return
