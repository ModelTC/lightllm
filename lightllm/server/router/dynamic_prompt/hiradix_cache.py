import torch
import torch.distributed as dist
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.log_utils import init_logger
from threading import Lock
from enum import Enum
from kvcache.python.jit import PyLocalCacheService
import time

logger = init_logger(__name__)


class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager, max_seq_length):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        logger.info("Initializing HiRadixCache")
        self.rank_in_node = rank_in_node
        try:
            # TODO: determine by model type && dp, tp
            store_once = True  # Deepseek -> True, Llama -> False
            self.do_store = store_once and self.rank_in_node == 0
            self.is_hi_radix_cache = True
            all_buffers = self.mem_manager.kv_buffer
            all_buffers = all_buffers.view(all_buffers.shape[0], all_buffers.shape[1], -1)
            self.py_cache_service = (
                PyLocalCacheService(
                    file="cache/cache_file",
                    storage_size=128 * (1024 ** 3),
                    num_shard=32,
                    kvcache_tensor=all_buffers,
                    num_worker=32,
                )
                if self.do_store
                else None
            )
            self.working_tasks = {}
        except Exception as e:
            logger.error(f"error alloc hi cache buffer {e}, fallback to normal radix cache")
            self.hi_cache_kv_buffer = None
            self.is_hi_radix_cache = False

    def insert_disk(self, req_id, key, value):
        if not self.do_store:
            return
        if req_id in self.working_tasks:
            self.abort_req_store_task(req_id)
        self.working_tasks[req_id] = self.py_cache_service.create(tokens=key, kv_page_indexer=value, mode="w")
        logger.info(f"Created store task for req {req_id}.")

    def abort_req_store_task(self, req_id):
        if not self.do_store or req_id not in self.working_tasks:
            return
        if self.working_tasks[req_id].ready():
            logger.info(f"Calling abort for req {req_id}, but is finished.")
            return
        logger.info(f"Aborting req {req_id} unfinished.")
        self.py_cache_service.az5(self.working_tasks[req_id])

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        pull_hi_cache_tensor = torch.tensor([0], dtype=torch.int64).cuda(self.rank_in_node)
        if self.do_store:
            tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
            max_len = self._query_hi_cache(key)  # x64
            logger.info(f"Matched {sum(len(s) for s in ans_value_list)} from gpu and {max_len} from disk.")
            pull_hi_cache_tensor[0] = max_len if (max_len > sum(len(s) for s in ans_value_list)) else 0
        dist.broadcast(pull_hi_cache_tensor, src=0)
        pull_hi_cache = False

        if pull_hi_cache_tensor[0] == 0 and not self.do_store:
            tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
        elif pull_hi_cache_tensor[0] > 0:
            pull_hi_cache = True
            max_len = pull_hi_cache_tensor[0]
            try:
                self.free_radix_cache_to_get_enough_token(max_len)
            except:
                logger.info(f"Unable to free on rank {self.rank_in_node}")
                pull_hi_cache_tensor[0] = 0
                pull_hi_cache = False
                ans_value_list = []
                tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        if pull_hi_cache:
            buffers = self.mem_manager.alloc(max_len)
            if self.do_store:
                read_task = self.py_cache_service.create(tokens=key[:max_len], kv_page_indexer=buffers, mode="r")
                while not read_task.ready():
                    time.sleep(0.05)
            dist.broadcast(self.mem_manager.get_index_kv_buffer(buffers)["kv_buffer"], src=0)
            logger.info(f"HiCache pulled one cache with len = {max_len}")
            self._insert_helper(self.root_node, key, buffers)
            ans_value_list = []
            tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        if tree_node != self.root_node:
            if len(ans_value_list) != 0:
                value = torch.concat(ans_value_list)
            else:
                assert False, "can not run to here"
            return tree_node, len(value), value
        else:
            self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _query_hi_cache(self, key) -> bool:
        query_result = self.py_cache_service.query(key)
        # query_result is a list of bool, find out the max len true continuous from start
        max_len = 0
        for result in query_result:
            if result:
                max_len += 1
            else:
                break
        return max_len * self.py_cache_service.tokens_per_block
