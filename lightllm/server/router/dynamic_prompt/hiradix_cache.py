import torch
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager
from lightllm.utils.log_utils import init_logger
from threading import Lock, Thread

logger = init_logger(__name__)

class HiRadixCache(RadixCache):
    def __init__(self, unique_name, total_token_num, rank_in_node, mem_manager, max_seq_length, py_cache_service):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        try:
            assert py_cache_service is not None
            self.py_cache_service = py_cache_service
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
        assert self.moving
        assert self.hi_cache_buffer_len > 0
        assert self.hi_cache_kv_buffer is not None
        key = self.hi_cache_key_buffer[:self.hi_cache_buffer_len].tolist()
        write_task = self.py_cache_service.create(tokens=key, kv_page_indexer=self.hi_cache_kv_buffer[:self.hi_cache_buffer_len], mode="w")
        while not write_task.ready(): 
            pass
        with self.moving_lock:
            self.moving = False

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        available_hi_result = self.cache_controller.readable_length(key)
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=False)
        if tree_node == self.root_node or available_hi_result > len(ans_value_list):
            hi_result = self.cache_controller.read(key)
            self._insert_helper(tree_node, key, hi_result)
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
