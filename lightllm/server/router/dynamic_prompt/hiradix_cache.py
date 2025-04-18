import torch
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
            self.is_hi_radix_cache = True
            all_buffers = self.mem_manager.kv_buffer
            all_buffers = all_buffers.view(all_buffers.shape[0], all_buffers.shape[1], -1)
            self.py_cache_service = PyLocalCacheService(
                file="cache/cache_file",
                storage_size=128 * (1024 ** 3),
                num_shard=32,
                kvcache_tensor=all_buffers,
                num_worker=32,
            )
            self.working_tasks = {}
        except Exception as e:
            logger.error(f"error alloc hi cache buffer {e}, fallback to normal radix cache")
            self.hi_cache_kv_buffer = None
            self.is_hi_radix_cache = False

    # write a new function, only insert input(after prefill), call after prefill,
    # then when the decode finishes, do syncronize to see whether this can be free
    # no buffer, parallel insert inputs
    def insert_disk(self, req_id, key, value):
        if req_id in self.working_tasks:
            self.wait_till_finish(req_id)
        self.working_tasks[req_id] = self.py_cache_service.create(tokens=key, kv_page_indexer=value, mode="w")
        logger.info(f"Created store task for req {req_id}.")

    def wait_till_finish(self, req_id):
        if req_id not in self.working_tasks:
            return
        starting_time = time.time()
        while not self.working_tasks[req_id].ready():
            time.sleep(0.01)
        logger.info(f"Waited {time.time() - starting_time}s for req {req_id}.")

    # def insert(self, key, value=None):
    #     if value is None:
    #         value = key

    #     assert len(key) == len(value)  # and len(key) >= 1
    #     if len(key) == 0:
    #         return 0

    #     # current implement is serial, TODO: make it parallel
    #     # if no hi_cache_buffer, work with normal radix cache
    #     if self.hi_cache_kv_buffer is not None:
    #         do_copy = False
    #         # and if is moving, ignore this insert request
    #         with self.moving_lock:
    #             if (not self.start_store_task) and self.write_task is not None:
    #                 if self.write_task.ready():
    #                     logger.info(f"HiCache of [{self.rank_in_node}]: stored len = {self.hi_cache_buffer_len}")
    #                     self.start_store_task = True # ensure ready => start new only one kvcache stores
    #                     do_copy = True
    #             elif self.write_task is None and self.starting:
    #                 self.starting = False
    #                 self.start_store_task = True
    #                 do_copy = True

    #         if do_copy:
    #             # copy the key and value to the hi_cache_buffer
    #             self.hi_cache_key_buffer[:len(key)].copy_(key)
    #             self.hi_cache_buffer_len = len(key)
    #             for buffer_index, index in enumerate(value):
    #                 kv_data = self.mem_manager.get_index_kv_buffer(index)
    #                 self.mem_manager.load_index_kv_buffer(self.hi_cache_kv_buffer[buffer_index], kv_data)
    #             # create a new thread to store the buffer
    #             self._store_buffer()

    #     return self._insert_helper(self.root_node, key, value)

    # def _store_buffer(self):
    #     logger.info(f"Storing buffer size = {self.hi_cache_buffer_len}")
    #     assert self.hi_cache_buffer_len > 0
    #     assert self.hi_cache_kv_buffer is not None
    #     key = self.hi_cache_key_buffer[:self.hi_cache_buffer_len].tolist()
    #     self.write_task = self.py_cache_service.create(
    #         tokens=key, kv_page_indexer=self.hi_cache_kv_buffer[:self.hi_cache_buffer_len], mode="w")
    #     with self.moving_lock:
    #         self.start_store_task = False

    def match_prefix(self, key, update_refs=False):
        st_time = time.time()
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        # add a parameter if get long enough (>50%)
        first_query_time = time.time()
        logger.info(f"HiCache of [{self.rank_in_node}]: No.1 First GPU query took {first_query_time - st_time}")
        max_len = self._query_hi_cache(key)  # x64
        hi_cache_query_time = time.time()
        logger.info(f"HiCache of [{self.rank_in_node}]: No.2 Disk query took {hi_cache_query_time - first_query_time}")
        logger.info(f"Matched {len(ans_value_list)} from gpu and {max_len} from disk.")
        pull_hi_cache = False
        if max_len > len(ans_value_list):
            pull_hi_cache = True
            try:
                self.free_radix_cache_to_get_enough_token(max_len)
            except:
                pull_hi_cache = False
        if pull_hi_cache:
            buffers = self.mem_manager.alloc(max_len)
            before_pull_time = time.time()
            logger.info(
                f"HiCache of [{self.rank_in_node}]: No.2.5 Before pull took {before_pull_time - hi_cache_query_time}"
            )
            read_task = self.py_cache_service.create(tokens=key[:max_len], kv_page_indexer=buffers, mode="r")
            while not read_task.ready():
                time.sleep(0.1)
            hicache_pull_time = time.time()
            logger.info(f"HiCache of [{self.rank_in_node}]: No.3 Disk pull took {hicache_pull_time - before_pull_time}")
            logger.info(f"HiCache pulled one cache with len = {max_len}")
            # maybe try: add a function to only insert middle part of kv cache
            self._insert_helper(self.root_node, key, buffers)
            insert_time = time.time()
            logger.info(f"HiCache of [{self.rank_in_node}]: No.4 Reinsert took {insert_time - hicache_pull_time}")
            ans_value_list = []
            tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
            logger.info(f"HiCache of [{self.rank_in_node}]: No.5 Re match prefix took {time.time() - insert_time}")
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
