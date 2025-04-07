import torch
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.log_utils import init_logger
from threading import Lock, Thread
from cache.ffi.pywarp import PyLocalCacheService

logger = init_logger(__name__)

class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager, max_seq_length):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        print(f"Initializing HiRadixCache")
        try:
            all_buffers = self.mem_manager.kv_buffer
            all_buffers = all_buffers.view(all_buffers.shape[0], all_buffers.shape[1], -1)
            self.py_cache_service = PyLocalCacheService(
                file="cache/cache_file", storage_size=32 * (1024**3),
                num_shard=32, kvcache=all_buffers, num_worker=32)
            self.hi_cache_buffer_len = 0
            self.hi_cache_key_buffer = torch.empty(max_seq_length, dtype=torch.int64, device="cpu")
            self.hi_cache_kv_buffer = self.mem_manager.alloc(max_seq_length)
            self.moving = False
            self.moving_lock = Lock()
        except Exception as e:
            logger.error(f"error alloc hi cache buffer {e}, fallback to normal radix cache")
            self.hi_cache_kv_buffer = None
    
    def insert(self, key, value=None):
        if value is None:
            value = key

        assert len(key) == len(value)  # and len(key) >= 1
        if len(key) == 0:
            return 0
        
        # current implement is serial, TODO: make it parallel
        # if no hi_cache_buffer, work with normal radix cache
        if self.hi_cache_kv_buffer is not None:
            do_copy = False
            # and if is moving, ignore this insert request
            with self.moving_lock:
                if not self.moving:
                    self.moving = True
                    do_copy = True
            if do_copy:
                # copy the key and value to the hi_cache_buffer
                self.hi_cache_key_buffer[:len(key)].copy_(key)
                self.hi_cache_buffer_len = len(key)
                for buffer_index, index in enumerate(value):
                    kv_data = self.mem_manager.get_index_kv_buffer(index)
                    self.mem_manager.load_index_kv_buffer(self.hi_cache_kv_buffer[buffer_index], kv_data)
                # create a new thread to store the buffer
                thread = Thread(target=self._store_buffer)
                thread.start()

        return self._insert_helper(self.root_node, key, value)
    
    def _store_buffer(self):
        logger.info(f"Storing buffer size = {self.hi_cache_buffer_len}")
        assert self.moving
        assert self.hi_cache_buffer_len > 0
        assert self.hi_cache_kv_buffer is not None
        key = self.hi_cache_key_buffer[:self.hi_cache_buffer_len].tolist()
        write_task = self.py_cache_service.create(tokens=key, kv_page_indexer=self.hi_cache_kv_buffer[:self.hi_cache_buffer_len].type(torch.int64).cuda(), mode="w")
        while not write_task.ready(): 
            pass
        logger.info(f"HiCache: stored one kvcache with len = {self.hi_cache_buffer_len}")
        with self.moving_lock:
            self.moving = False

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        use_hi_cache = self._query_hi_cache(key, len(ans_value_list))
        if use_hi_cache:
            pull_hi_cache = True
            try:
                self.free_radix_cache_to_get_enough_token(len(key))
            except:
                pull_hi_cache = False
        if pull_hi_cache:
            buffers = self.mem_manager.alloc(len(key)).type(torch.int64).cuda()
            read_task = self.py_cache_service.create(tokens=key, kv_page_indexer=buffers, mode="r")
            while not read_task.ready():
                pass
            logger.info(f"HiCache pulled one cache with len = {len(key)}")
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
    
    def _query_hi_cache(self, key, gpu_ans_len) -> bool:
        query_result = self.py_cache_service.query(key)
        # query_result is a list of bool, find out the max len true continuous from start
        max_len = 0
        for result in query_result:
            if result:
                max_len += 1
            else:
                break
        return max_len > gpu_ans_len