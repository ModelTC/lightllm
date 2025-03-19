import torch
from .cache_controller import HiCacheController
from .radix_cache import RadixCache, TreeNode, match
from typing import Tuple, Dict, Set, List
from lightllm.common.mem_manager import MemoryManager


class HiRadixCache(RadixCache):
    def __init__(self, cache_controller: HiCacheController, unique_name, total_token_num, rank_in_node, mem_manager: MemoryManager = None):
        super().__init__(unique_name, total_token_num, rank_in_node, mem_manager)
        self.cache_controller = cache_controller
    
    def insert(self, key, value=None):
        if value is None:
            value = key

        assert len(key) == len(value)  # and len(key) >= 1
        if len(key) == 0:
            return 0
        
        self.cache_controller.write(key, value)
        return self._insert_helper(self.root_node, key, value)
    
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
