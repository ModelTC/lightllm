import torch
import torch.distributed as dist
import rpyc
from typing import Dict, List, Tuple, Optional
from rpyc.utils.classic import obtain
from .decode_impl import ContinuesBatchBackendForDecodeNode
from lightllm.common.basemodel.infer_lock import acquire_lock_until_ready, release_acquired_lock
from .decode_task_cache import g_kv_move_task_cache
from lightllm.server.pd_io_struct import KVMoveTask
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

    def exposed_alloc_to_frozen_some_tokens(self, move_task: KVMoveTask) -> Optional[int]:
        move_task = obtain(move_task)
        acquire_lock_until_ready()
        try:
            key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
            tree_node, kv_len, fused_token_indexes = self.backend.radix_cache.match_prefix(key, update_refs=True)
            # 如果没匹配到，说明长度是0， 将fused_token_indexes做一下转换
            fused_token_indexes = [] if fused_token_indexes is None else fused_token_indexes.tolist()
            need_len = len(move_task.input_tokens) - kv_len
            if need_len == 0:
                alloc_token_indexes = []
            else:
                alloc_token_indexes = self.backend.model.mem_manager.alloc(need_len).detach().cpu().tolist()

            if alloc_token_indexes is None:
                self.backend.radix_cache.dec_node_ref_counter(tree_node)
                return None

            move_task.decode_token_indexes = alloc_token_indexes
            move_task.move_kv_len = need_len

            g_kv_move_task_cache[move_task.group_request_id] = (move_task, tree_node, fused_token_indexes)
            return move_task.decode_token_indexes
        finally:
            release_acquired_lock()

    def exposed_put_kv_received_to_radix_cache(self, group_req_id: int):
        group_req_id = obtain(group_req_id)
        acquire_lock_until_ready()
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        radix_cache = self.backend.radix_cache
        key = torch.tensor(move_task.input_tokens, dtype=torch.int64, device="cpu")
        value = torch.tensor(fused_token_indexes + move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        prefix_len = radix_cache.insert(key, value)
        assert len(fused_token_indexes) <= prefix_len
        self.backend.model.mem_manager.free(value[len(fused_token_indexes) : prefix_len].cuda())
        self.backend.radix_cache.dec_node_ref_counter(tree_node)

        # 申请一段key，把radix cache 锁住，防止极端情况下被刷掉, decode 端通过减两次引用计数来修正。
        _, kv_len, _ = self.backend.radix_cache.match_prefix(key, update_refs=True)
        assert len(key) == kv_len
        release_acquired_lock()
        return

    def exposed_fail_to_realese_forzen_tokens(self, group_req_id: int):
        group_req_id = obtain(group_req_id)
        acquire_lock_until_ready()
        move_task, tree_node, fused_token_indexes = g_kv_move_task_cache.pop(group_req_id)
        value = torch.tensor(move_task.decode_token_indexes, dtype=torch.int64, device="cpu")
        self.backend.model.mem_manager.free(value.cuda())
        self.backend.radix_cache.dec_node_ref_counter(tree_node)
        release_acquired_lock()
        return

    def exposed_put_mem_manager_to_mem_queue(self):
        self.backend.mem_queue.put(self.backend.model.mem_manager)
        logger.info("put mem manager to info_queues ok")
        return
