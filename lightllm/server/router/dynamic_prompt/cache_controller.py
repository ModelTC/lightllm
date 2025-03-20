import torch
import threading
import time
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from threading import Thread, Event, RLock
from queue import Queue
from enum import Enum
from lightllm.common.mem_manager import MemoryManager

BLOCK_SIZE = 16384

def get_torch_tensor_size(tensor: torch.Tensor):
    return tensor.nelement() * tensor.element_size()

class LoadStatus(Enum):
    UNLOADED = 0
    LOADING = 1
    LOADED = 2

class CacheNode:
    def __init__(self, parent=None, split_token_idx=None):
        self.parent = parent  # 父节点
        self.split_token_idx = split_token_idx  # 从父节点分裂的位置
        self.children = {}  # (token_id, split_position) -> (child_node, split_position)
        self.cache_indices = []  # 存储kv cache在mem_manager中的索引
        self.token_ids = []  # 当前节点存储的token ids
        self.hash = None  # 存储在磁盘上的唯一标识
        self.status = LoadStatus.UNLOADED  # 加载状态

class HiCacheController:
    def __init__(self, mem_manager: MemoryManager):
        self.mem_manager = mem_manager
        self.service = None  # 将由外部代码初始化
        
        self.root = CacheNode()
        self.root.hash = "root_" + str(time.time())
        
        self.node_cache = {self.root.hash: self.root}  # hash -> node
        
        self.token_kvcache_size = None  # 每个token的kvcache大小
        
        self.node_lock = RLock()
        
        # 添加写任务队列
        self.writetaskqueue = Queue()
        self.write_thread_running = True
        
        # 启动处理写任务的线程
        self.write_thread = Thread(target=self._process_write_tasks)
        self.write_thread.daemon = True
        self.write_thread.start()
    
    def store_mem(self, indices, value):
        for idx, value_dim in zip(indices, range(value.shape[0])):
            self.mem_manager.load_index_kv_buffer(idx, {"kv_buffer": value[value_dim]})
        return indices
    
    def get_mem(self, indices):
        if len(indices) == 0:
            return torch.tensor([])
        return torch.stack([self.mem_manager.get_index_kv_buffer(idx)["kv_buffer"] for idx in indices], dim=0)
    
    def reset(self):
        """重置缓存控制器"""
        # 停止写任务线程
        self.write_thread_running = False
        self.write_thread.join(timeout=1)
        
        self.root = CacheNode()
        self.root.hash = "root_" + str(time.time())
        self.node_cache = {self.root.hash: self.root}
        
        # 重新创建队列和启动线程
        self.writetaskqueue = Queue()
        self.write_thread_running = True
        self.write_thread = Thread(target=self._process_write_tasks)
        self.write_thread.daemon = True
        self.write_thread.start()
    
    def _ensure_node_loaded(self, node_hash):
        """确保节点已加载到内存中"""
        assert node_hash in self.node_cache, f"Node {node_hash} not found in cache"
        assert node_hash[:4] != "root", "Cannot load root node"
        with self.node_lock:
            if self.node_cache[node_hash].status == LoadStatus.LOADED:
                return
            if self.node_cache[node_hash].status == LoadStatus.LOADING:
                while self.node_cache[node_hash].status != LoadStatus.LOADED:
                    time.sleep(0.01)
                return
            if self.node_cache[node_hash].status == LoadStatus.UNLOADED:
                self.node_cache[node_hash].status = LoadStatus.LOADING
        
        task = self.service.create(hashs=[node_hash], mode="r")
        self.service.commit(task)
        # 需要等待节点加载完成
        while not task.ready():
            time.sleep(0.01)
        for node_hash, node_data in zip(task.hashs, task.data):
            assert node_hash in self.node_cache
            node = self.node_cache[node_hash]
            node.cache_indices = self.store_mem(node.cache_indices, node_data)
            print(f"Node {node_hash} loaded with {len(node.cache_indices)} cache indices")
        print(f"Node {node_hash} loaded to memory")
    
    def _persist_node(self, node):
        """将节点持久化到磁盘"""
        if not node.hash:
            # 为新节点生成hash
            node.hash = f"node_{id(node)}_{time.time()}"
        
        print(f"Persisting node {node.hash} with {len(node.token_ids)} tokens")
        
        task = self.service.create(hashs=[node.hash], value=self.get_mem(node.cache_indices), mode="w")
        self.service.commit(task)
        self.node_cache[node.hash] = node
    
    def write(self, key: torch.Tensor, value: torch.Tensor):
        """
        将写任务加入队列，由后台线程异步处理
        
        key: token_ids序列
        value: 对应的KV缓存索引
        """
        # 将任务加入队列
        self.writetaskqueue.put((key.clone(), value.clone()))
    
    def _process_write_tasks(self):
        """后台线程处理写任务队列"""
        while self.write_thread_running:
            if not self.writetaskqueue.empty():
                # 从队列获取任务
                key, value = self.writetaskqueue.get()
                # 执行实际的写操作
                self._do_write(key, value)
                self.writetaskqueue.task_done()
            else:
                # 队列为空时短暂休眠，避免CPU占用过高
                time.sleep(0.01)
    
    def _do_write(self, key: torch.Tensor, value: torch.Tensor):
        """
        实际执行写入token序列及其对应的KV缓存索引
        
        key: token_ids序列
        value: 对应的KV缓存索引
        """
        token_ids = key.cpu().tolist()
        indices = value.cpu().tolist()
        
        # 首次计算每个token的kvcache大小
        if self.token_kvcache_size is None:
            kvcache = self.get_mem(indices[:1])  # 计算单个token的kvcache
            self.token_kvcache_size = get_torch_tensor_size(kvcache)
            print(f"Single token KV cache size: {self.token_kvcache_size} bytes, Block size: {BLOCK_SIZE}")
        
        current = self.root
        position = 0
        relative_position = 0
        
        while position < len(token_ids):
            token_id = token_ids[position]
            print(f"Writing token {token_id} at position {position}, current node has {len(current.token_ids)} tokens")
            print(f"relative_position: {relative_position}, node_hash: {current.hash}")
            child_key = (token_id, relative_position)
            
            if child_key in current.children:
                print(f"Child key {child_key} found in current.children")
                child_info = current.children[child_key]
                assert isinstance(child_info[0], CacheNode)
                child_hash = child_info[0].hash
                self._ensure_node_loaded(child_hash)
                current = self.node_cache[child_hash]
                position += 1
                relative_position = 0 # next time relative pos is 0
            else:
                # 计算当前节点剩余空间
                remaining_space = BLOCK_SIZE - len(current.cache_indices) * self.token_kvcache_size
                
                # root 不应存储任何内容
                if self.token_kvcache_size <= remaining_space and current != self.root:
                    # 当前节点有足够空间
                    current.token_ids.append(token_ids[position])
                    current.cache_indices.append(indices[position])
                    position += 1
                    relative_position += 1
                    self._persist_node(current)
                else:
                    # 当前节点已满，需要创建新节点
                    new_node = CacheNode(parent=current, split_token_idx=len(current.token_ids))
                    print(f"Creating new node at split position {new_node.split_token_idx}, parent hash: {current.hash}")
                    
                    # 将token添加到新节点
                    new_node.token_ids.append(token_ids[position])
                    new_node.cache_indices.append(indices[position])
                    position += 1
                    relative_position = 0 # next time relative pos is 0, not affecting child_key
                    
                    # 建立父子关系
                    current.children[(token_id, new_node.split_token_idx)] = (new_node, new_node.split_token_idx)
                    
                    # 持久化
                    self._persist_node(new_node)
                    self._persist_node(current)
                    
                    current = new_node
        
        # 确保最后修改的节点被持久化
        self._persist_node(current)
    
    def readable_length(self, key: torch.Tensor) -> int:
        """
        计算key对应的KV缓存索引可读取的长度
        """
        token_ids = key.cpu().tolist()
        current = self.root
        position = 0
        relative_position = 0
        readable_count = 0
        
        while position < len(token_ids):
            token_id = token_ids[position]
            
            # 检查当前节点的token
            if relative_position < len(current.token_ids) and current.token_ids[relative_position] == token_id:
                readable_count += 1
                position += 1
                relative_position += 1
                continue
            
            # 查找子节点
            child_key = (token_id, relative_position)
            if child_key in current.children:
                child_info = current.children[child_key]
                assert isinstance(child_info[0], CacheNode)
                child_hash = child_info[0].hash
                current = self.node_cache[child_hash]
                relative_position = 0
            else:
                # 未找到匹配的路径，返回已读取的长度
                return readable_count
        
        return readable_count
    
    def read(self, key: torch.Tensor) -> torch.Tensor:
        """
        读取token序列对应的KV缓存索引
        key: token_ids序列
        返回: 对应的KV缓存索引
        """
        print(f"Reading key: {key}")
        token_ids = key.cpu().tolist()
        result_indices = []
        
        current = self.root
        position = 0
        relative_position = 0
        
        while position < len(token_ids):
            token_id = token_ids[position]
            print(f"Reading token {token_id} at position {position}, node total {len(current.token_ids)} tokens from node hash {current.hash}")
            
            # 检查当前节点的token
            if relative_position < len(current.token_ids) and current.token_ids[relative_position] == token_id:
                # TODO: 将读到的东西存到 result_indices 中
                result_indices.append(current.cache_indices[relative_position])
                position += 1
                relative_position += 1
                continue
            
            # 查找子节点
            child_key = (token_id, relative_position)
            print(f"Looking for child {child_key} in node {current.hash}: {current.children}")
            if child_key in current.children:
                child_info = current.children[child_key]
                assert isinstance(child_info[0], CacheNode)
                child_hash = child_info[0].hash
                self._ensure_node_loaded(child_hash)
                current = self.node_cache[child_hash]
                relative_position = 0
            else:
                # 未找到匹配的路径
                return torch.tensor(result_indices)
        
        return torch.tensor(result_indices)


class HiHostTask:
    def __init__(self, hashs, mode, value=None):
        self.hashs = hashs
        self.mode = mode
        self._ready = Event()
        self.data = value
    
    def ready(self):
        return self._ready.is_set()
    
    def set_ready(self):
        self._ready.set()

class HiHostService:
    def __init__(self):
        self.tasks = Queue()
        self.added_count = 0
        self.finished_count = 0
        self.running = True
        self.hash_data = {} # hash -> (data, device)
        self.worker = Thread(target=self.process_tasks)
        self.worker.daemon = True
        self.worker.start()
    
    def process_tasks(self):
        while self.running:
            if not self.tasks.empty():
                start_time = time.time()
                task = self.tasks.get()
                self.complete(task)
                task.set_ready()
                print(f"Task for {task.hashs} completed after {time.time() - start_time:.2f}s")
            else:
                time.sleep(0.01)
    
    def complete(self, task):
        if task.mode == "r":
            assert all(hash in self.hash_data for hash in task.hashs)
            task.data = torch.stack(list(self.hash_data[hash][0] for hash in task.hashs))
            task.data.to(self.hash_data[task.hashs[0]][1])
        elif task.mode == "w":
            device = task.data[0].device
            for hash, value in zip(task.hashs, task.data):
                self.hash_data[hash] = (value.to("cpu"), device)
        self.finished_count += 1
    
    def create(self, hashs, mode, value=None):
        assert mode in ["r", "w"]
        if not isinstance(value, list):
            value = [value]
        assert len(value) == len(hashs)
        task = HiHostTask(hashs, mode, value)
        return task
    
    def all_finished(self):
        return self.tasks.empty() and self.added_count == self.finished_count
    
    def wait_till_all_finished(self):
        while not self.all_finished():
            time.sleep(0.01)
    
    def commit(self, task):
        self.tasks.put(task)
        self.added_count += 1
    
    def shutdown(self):
        self.running = False
        self.worker.join()

