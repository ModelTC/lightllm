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
            # TODO: determine by model type && dp, tp
            store_once = True  # Deepseek -> True, Llama -> False
            self.do_store = store_once and self.rank_in_node == 0
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
        if not self.do_store:
            return
        if req_id in self.working_tasks:
            self.abort_req_store_task(req_id)
        self.working_tasks[req_id] = self.py_cache_service.create(tokens=key, kv_page_indexer=value, mode="w")
        logger.info(f"Created store task for req {req_id}.")

    def abort_req_store_task(self, req_id):
        if not self.do_store:
            return
        if self.working_tasks[req_id].ready():
            logger.info(f"Calling abort for req {req_id}, but is finished.")
            return
        logger.info(f"Aborting req {req_id} unfinished.")
        self.py_cache_service.az5(self.working_tasks[req_id])

    # TODO: finish this function to only update new ones
    def _reinsert_helper(self, node: TreeNode, key, value, ans_value_list: list, update_refs=False):
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        if update_refs:
            node.ref_counter += 1
            # from 0 to 1 need update refs token num
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        try:
            if len(key) == 0:
                return node

            first_key_id = key[0].item()
            if first_key_id in node.children.keys():
                child: TreeNode = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
                if prefix_len == len(key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    child.update_time()
                    ans_value_list.append(child.token_mem_index_value)
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len

                elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    split_parent_node = child.split_node(prefix_len)
                    new_node = split_parent_node.add_and_return_new_child(key, value)
                    # update total token num
                    self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                    if new_node.is_leaf():
                        self.evict_tree_set.add(new_node)

                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len
                elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                    return prefix_len + self._insert_helper(child, key[prefix_len:], value[prefix_len:])
                else:
                    assert False, "can not run to here"

            else:
                new_node = node.add_and_return_new_child(key, value)
                # update total token num
                self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                ans_value_list.append(new_node.token_mem_index_value)
                if update_refs:
                    new_node.ref_counter += 1
                    if new_node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)
                return new_node
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

    def match_prefix(self, key, update_refs=False):
        st_time = time.time()
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
        # add a parameter if get long enough (>50%)
        first_query_time = time.time()
        logger.info(f"HiCache of [{self.rank_in_node}]: No.1 First GPU query took {first_query_time - st_time}")
        max_len = self._query_hi_cache(key)  # x64
        hi_cache_query_time = time.time()
        logger.info(f"HiCache of [{self.rank_in_node}]: No.2 Disk query took {hi_cache_query_time - first_query_time}")
        logger.info(f"Matched {sum(len(s) for s in ans_value_list)} from gpu and {max_len} from disk.")
        pull_hi_cache = False
        if max_len > sum(len(s) for s in ans_value_list):
            pull_hi_cache = True
            try:
                self.free_radix_cache_to_get_enough_token(max_len)
            except:
                if update_refs:
                    tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
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
            logger.info(
                f"HiCache of [{self.rank_in_node}]: No.5 Re match prefix took {time.time() - insert_time}"
                + f" matched {sum(len(s) for s in ans_value_list)} tokens"
            )
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
