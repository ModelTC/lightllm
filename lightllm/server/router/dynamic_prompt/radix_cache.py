#Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple
from sortedcontainers import SortedSet
class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter
    
time_gen = UniqueTimeIdGenerator()

class TreeNode:
    def __init__(self):
        self.children = {} #这里的键 为 token_id_key 的第一个元素
        self.parent = None
        self.token_id_key = None
        self.token_mem_index_value = None # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id() # 用于标识时间周期

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)
    

    def split_node(self, prefix_len):
        new_node = TreeNode()
        new_node.parent = self
        new_node.token_id_key = self.token_id_key[prefix_len:]
        new_node.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        new_node.children = self.children
        new_node.ref_counter = self.ref_counter

        self.token_id_key = self.token_id_key[0:prefix_len]
        self.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        self.children = {}
        self.children[new_node.token_id_key[0]] = new_node
        return new_node
    

    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0]
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self
        return child
    
    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0



def match(key, seq):
    i = 0
    for k, w in zip(key, seq):
        if k != w:
            break
        i += 1
    return i


class RadixCache:
    def __init__(self):
        self.root_node = {}
        self.root_node.token_id_key = []
        self.root_node.token_mem_index_value = []
        self.root_node.ref_counter = 0

        self.evict_tree_set = SortedSet(key=lambda x: x.get_compare_key()) # 自定义比较器
        self.evict_tree_set.add(self.root_node)

    
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
                    assert False, "cannot run to here"

            else:
                new_node = node.add_and_return_new_child(key, value)
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

        
# ///////////////////////////////////////////////////////////////////////////////


    def match_prefix(self, key):
        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            value = torch.concat(value)
        return value, last_node[0]

    def insert(self, key, value=None):
        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens, evict_callback):
        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.ref_counter > 0:
                continue

            num_evicted += evict_callback(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.ref_counter += 1
            node = node.parent
        return delta

    def dec_ref_counter(self, node):
        delta = 0
        while node != self.root_node:
            if node.ref_counter == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.ref_counter -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####
    def _match_prefix_helper(self, node, key, value, last_node):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)
            if prefix_len != 0:
                if prefix_len < len(c_key):
                    new_node = self._split_node(c_key, child, prefix_len)
                    value.append(new_node.value)
                    last_node[0] = new_node
                else:
                    value.append(child.value)
                    last_node[0] = child
                    self._match_prefix_helper(child, key[prefix_len:], value, last_node)
                break

    def _split_node(self, key, child, split_len):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:]: child}
        new_node.parent = child.parent
        new_node.ref_counter = child.ref_counter
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len]] = new_node
        del new_node.parent.children[key]
        return new_node

    def _insert_helper(self, node, key, value):
        node.last_access_time = time.time()

        for c_key, child in node.children.items():
            prefix_len = match(c_key, key)

            if prefix_len == len(c_key):
                if prefix_len == len(key):
                    return prefix_len
                else:
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            if prefix_len:
                new_node = self._split_node(c_key, child, prefix_len)
                return prefix_len + self._insert_helper(
                    new_node, key[prefix_len:], value[prefix_len:]
                )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.value = value
            node.children[key] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node, indent):
        for key, child in node.children.items():
            print(" " * indent, len(key), key[:10], f"r={child.ref_counter}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(k)

    def _total_size_helper(self, node):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list


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