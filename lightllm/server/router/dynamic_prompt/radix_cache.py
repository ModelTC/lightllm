#Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import heapq
import time
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
from sortedcontainers import SortedSet
from .shared_arr import SharedArray, SharedIdxNode, SharedTreeIdxManager
class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter
    
time_gen = UniqueTimeIdGenerator()

class TreeNode:
    def __init__(self, shared_idx_manager : SharedTreeIdxManager):
        self.shared_idx_manager = shared_idx_manager
        self.children = {} #这里的键 为 token_id_key 的第一个元素
        self.parent: TreeNode = None
        self.token_id_key = None
        self.token_mem_index_value = None # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.shared_idx_node: SharedIdxNode = self.shared_idx_manager.alloc()
        self.time_id = time_gen.generate_time_id() # 用于标识时间周期

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)
    
    def split_node(self, prefix_len):
        new_node = TreeNode(self.shared_idx_manager)
        new_node.parent = self
        new_node.token_id_key = self.token_id_key[prefix_len:]
        new_node.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        new_node.children = self.children
        new_node.ref_counter = self.ref_counter

        self.token_id_key = self.token_id_key[0:prefix_len]
        self.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        self.children = {}
        self.children[new_node.token_id_key[0]] = new_node

        # 更新shared 信息
        self.shared_idx_node.set_node_value_len(prefix_len)
        self.shared_idx_node.set_node_prefix_total_len(self.get_parent_prefix_total_len() + prefix_len)
        
        new_node.shared_idx_node.set_parent_idx(self.shared_idx_node.get_idx())
        new_len = len(new_node.token_mem_index_value)
        new_node.shared_idx_node.set_node_value_len(new_len)
        new_node.shared_idx_node.set_node_prefix_total_len(new_node.get_parent_prefix_total_len() + new_len)
        return new_node
    
    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode(self.shared_idx_manager)
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0]
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self

        # 更新shared 信息
        child.shared_idx_node.set_parent_idx(self.shared_idx_node.get_idx())
        new_len = len(child.token_mem_index_value)
        child.shared_idx_node.set_node_value_len(new_len)
        child.shared_idx_node.set_node_prefix_total_len(child.get_parent_prefix_total_len() + new_len)
        return child
    
    def remove_child(self, child_node):
        del self.children[child_node.token_id_key[0]]
        child_node.parent = None
        return
    
    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0
    
    def get_parent_prefix_total_len(self):
        return self.parent.shared_idx_node.get_node_prefix_total_len()

    def __del__(self):
        self.shared_idx_manager.free(self.shared_idx_node.get_idx())


def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self, total_token_num, tp_id):
        self.shared_idx_manager = SharedTreeIdxManager(total_token_num, tp_id)
        self.root_node = TreeNode(self.shared_idx_manager)
        self.root_node.token_id_key = []
        self.root_node.token_mem_index_value = []
        self.root_node.ref_counter = 1 # 初始化为 1 保证永远不会被 evict 掉

        self.evict_tree_set = SortedSet(key=lambda x: x.get_compare_key()) # 自定义比较器
        self.evict_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(f"tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)

    
    def insert(self, key, value):
        assert len(key) == len(value) and len(key) >= 1
        return self._insert_helper(self.root_node, key, value)
    
    def _insert_helper(self, node: TreeNode, key, value):
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        try:
            first_key_id = key[0]
            if first_key_id in node.children.keys():
                child : TreeNode = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)                    
                if prefix_len == len(key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len
                
                elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    split_new_node = child.split_node(prefix_len)
                    new_node = child.add_and_return_new_child(key, value)
                    # update total token num
                    self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
                    
                    if split_new_node.is_leaf():
                        self.evict_tree_set.add(split_new_node)
                    if new_node.is_leaf():
                        self.evict_tree_set.add(new_node)

                    child.update_time()
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
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)
                return 0
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

    def match_prefix(self, key, update_refs = False):
        assert len(key) != 0
        ans_value_list = []
        self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        if len(ans_value_list) != 0:
            value = torch.concat(ans_value_list)
        else:
            value = [] # 应该返回一个 0 shape tensor 更好
        return len(value), value

    def _match_prefix_helper(self, node: TreeNode, key, ans_value_list: list, update_refs=False):
        if node.is_leaf():
            self.evict_tree_set.discard(node)
        
        if update_refs:
            node.ref_counter += 1
            # from 0 to 1 need update refs token num
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        
        try:
            if len(key) == 0:
                return
        
            first_key_id = key[0]
            if first_key_id not in node.children.keys():
                return
            else:
                child = node.children[first_key_id]
                prefix_len = match(key, child.token_id_key)
                if prefix_len == len(child.token_id_key):
                    ans_value_list.append(child.token_mem_index_value)
                    self._match_prefix_helper(child, key[prefix_len:], ans_value_list, update_refs=update_refs)
                elif prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    new_split_node = child.split_node(prefix_len)
                    ans_value_list.append(child.token_mem_index_value)
                    
                    if update_refs:
                        child.ref_counter += 1
                        # from 0 to 1 need update refs token num
                        if child.ref_counter == 1:
                            self.refed_tokens_num.arr[0] += len(child.token_mem_index_value)

                    child.update_time()
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    if new_split_node.is_leaf():
                        self.evict_tree_set.add(new_split_node)
                    
                    return
                else:
                    assert False, "error state"
        finally:
            node.update_time()
            if node.is_leaf():
                self.evict_tree_set.add(node)

    def evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        while num_evicted < need_remove_tokens:
            node : TreeNode = self.evict_tree_set.pop(0)
            assert node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node, "error evict tree node state"
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node : TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)
        return
    
    def dec_node_ref_counter(self, node):
        while node != self.root_node:
            if node.ref_counter == 1:
                self.refed_tokens_num -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent
        return
    
    def print_self(self, indent=0):
        self._print_helper(self, self.root_node, indent)
    
    def _print_helper(self, node: TreeNode, indent):
        for first_token_id, child in node.children.items():
            print(" " * indent,  f"fk: {first_token_id} k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} time_id: {node.time_id} prefix_total_len: {node.shared_idx_node.get_node_prefix_total_len()}")
            self._print_helper(child, indent=indent + 2)
        return


class RadixCacheReadOnlyClient:
    """
    router 端只读用的客户端，用于从共享内存中读取树结构中的信息，用于进行prompt cache 的调度等
    """
    def __init__(self, total_token_num, tp_id):
        self.shared_idx_manager = SharedTreeIdxManager(total_token_num, tp_id)
        self.refed_tokens_num = SharedArray(f"refed_tokens_num_{tp_id}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(f"tree_total_tokens_num_{tp_id}", (1,), dtype=np.int64)
    
    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]
    
    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]
    
    def get_shared_node(self, idx):
        return self.shared_idx_manager.get_shared_node(idx)
    
    def get_all_parent_shared_nodes(self, idx):
        node = self.shared_idx_manager.get_shared_node(idx)
        ans_list = [node]
        while node.get_parent_idx() != -1:
            node = self.shared_idx_manager.get_shared_node(node.get_parent_idx())
            ans_list.append(node)
        return ans_list

    

# ///////////////////////////////////////////////////////////////////////////////

if __name__ == "__main__":
    tree = RadixCache()

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()