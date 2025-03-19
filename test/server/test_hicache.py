# test_hicache.py
import torch
import time
import random
from threading import Thread, Event
from queue import Queue
from lightllm.server.router.dynamic_prompt.cache_controller import HiCacheController, CacheNode, BLOCK_SIZE

class MockMemoryManager:
    """模拟内存管理器，仅返回连续的索引值"""
    def __init__(self):
        self.current_idx = 0
        self.kvcache_store = {}

    def alloc(self, size):
        indices = list(range(self.current_idx, self.current_idx + size))
        self.current_idx += size
        self.store(indices, torch.tensor([[random.randint(0, 0xffff) for __ in range(512)] for _ in range(size)]))
        return indices
    
    def to_kvcache(self, indices):
        assert all([idx in self.kvcache_store for idx in indices]), f"Not all of {indices} are not found in kvcache_store"
        return torch.tensor([self.kvcache_store[idx].tolist() for idx in indices])
    
    def store(self, indices, value):
        print(f"[TEST:MemManager] Storing {value.shape} at {indices}")
        for idx, value_dim in zip(indices, range(value.shape[0])):
            self.kvcache_store[idx] = value[value_dim]
            print(f"[TEST:MemManager] Stored {value[value_dim].shape} at {idx}")
        return indices
    
    def exist(self, indices):
        return all([idx in self.kvcache_store for idx in indices])
    
    def free(self, indices):
        print(f"[TEST:MemManager] Freeing {indices}")
        for idx in indices:
            del self.kvcache_store[idx]

class MockTask:
    def __init__(self, hashs, mode, value=None):
        self.hashs = hashs
        self.mode = mode
        self._ready = Event()
        self.data = value
    
    def ready(self):
        return self._ready.is_set()
    
    def set_ready(self):
        self._ready.set()

class MockService:
    def __init__(self):
        self.tasks = Queue()
        self.added_count = 0
        self.finished_count = 0
        self.running = True
        self.hash_data = {}
        self.worker = Thread(target=self.process_tasks)
        self.worker.daemon = True
        self.worker.start()
    
    def process_tasks(self):
        while self.running:
            if not self.tasks.empty():
                # 模拟随机延迟后完成任务
                delay = random.uniform(0.01, 0.1)
                time.sleep(delay)
                task = self.tasks.get()
                self.complete(task)
                task.set_ready()
                print(f"Task for {task.hashs} completed after {delay:.2f}s")
            else:
                time.sleep(0.01)
    
    def complete(self, task):
        if task.mode == "r":
            assert all(hash in self.hash_data for hash in task.hashs)
            task.data = torch.stack(list(self.hash_data[hash] for hash in task.hashs))
        elif task.mode == "w":
            for hash, value in zip(task.hashs, task.data):
                self.hash_data[hash] = value
        self.finished_count += 1
    
    def create(self, hashs, mode, value=None):
        assert mode in ["r", "w"]
        if not isinstance(value, list):
            value = [value]
        assert len(value) == len(hashs)
        task = MockTask(hashs, mode, value)
        self.tasks.put(task)
        self.added_count += 1
        return task
    
    def all_finished(self):
        return self.tasks.empty() and self.added_count == self.finished_count
    
    def wait_till_all_finished(self):
        time.sleep(1)
        while not self.all_finished():
            time.sleep(0.01)
    
    def commit(self, task):
        pass  # 在Mock中不需要实现
    
    def shutdown(self):
        self.running = False
        self.worker.join()

def setup():
    mem_manager = MockMemoryManager()
    service = MockService()
    hicache = HiCacheController(mem_manager)
    hicache.service = service  # 注入模拟服务
    
    indices = mem_manager.alloc(5)
    print(mem_manager.to_kvcache(indices))
    
    # 预先计算单token大小
    dummy_indices = mem_manager.alloc(1)
    kvcache = mem_manager.to_kvcache(dummy_indices[:1])
    token_size = kvcache.nelement() * kvcache.element_size()
    print(f"[TEST] Single token KV cache size: {token_size} bytes, Block size: {BLOCK_SIZE}")
    
    return mem_manager, service, hicache, token_size

def test_basic_write_read(mem_manager, hicache, token_size):
    # 计算每个块可容纳的token数量
    tokens_per_block = BLOCK_SIZE // token_size
    print(f"[TEST] Each block can hold {tokens_per_block} tokens")
    
    # 生成测试数据：刚好占满一个块
    token_ids = list(range(tokens_per_block))
    indices = mem_manager.alloc(len(token_ids))
    kvcache = mem_manager.to_kvcache(indices)
    print(f"[TEST] Generated KV cache with shape: {kvcache.shape}, type: {kvcache.dtype}")
    
    # 写入缓存
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    
    # 等待任务完成
    hicache.service.wait_till_all_finished()
    
    mem_manager.free(indices)
    
    # 读取验证
    result = hicache.read(torch.tensor(token_ids))
    result = mem_manager.to_kvcache(result.tolist())
    assert result.eq(kvcache).all(), f"Retrieved kvcache: {result}, Expected kvcache: {kvcache}"
    print(f"[TEST] Basic test passed. Retrieved kvcache\n\n")

def test_node_splitting(mem_manager, hicache, token_size):
    tokens_per_block = BLOCK_SIZE // token_size
    # 生成超过一个块的数据
    token_ids = list(range(12, 12 + tokens_per_block * 3 + 1))
    indices = mem_manager.alloc(len(token_ids))
    kvcache = mem_manager.to_kvcache(indices)
    
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    hicache.service.wait_till_all_finished()
    
    # 验证根节点应该有子节点
    root = hicache.root
    assert len(root.children) > 0
    print(f"\nRoot node has {len(root.children)} children")
    
    # 读取完整序列
    result = hicache.read(torch.tensor(token_ids))
    result = mem_manager.to_kvcache(result.tolist())
    assert result.eq(kvcache).all(), f"Retrieved kvcache: {result}, Expected kvcache: {kvcache}"
    print(f"[TEST] Node splitting test passed. Retrieved kvcache: {result.shape}\n\n")

def test_partial_read(mem_manager, hicache):
    token_ids = [97, 98, 99, 100, 101, 102]
    indices = mem_manager.alloc(len(token_ids))
    kvcache = mem_manager.to_kvcache(indices)
    hicache.write(torch.tensor(token_ids), torch.tensor(indices))
    hicache.service.wait_till_all_finished()
    
    # 查询存在的部分前缀
    result = hicache.read(torch.tensor([97, 98, 99]))
    result = mem_manager.to_kvcache(result.tolist())
    assert result.eq(kvcache[:3]).all()
    print(f"[TEST] Partial read passed")
    
    # 查询不存在的前缀
    result = hicache.read(torch.tensor([97, 98, 100]))
    assert len(result) == 2
    result = mem_manager.to_kvcache(result.tolist())
    assert result.eq(kvcache[:2]).all()
    print(f"[TEST] Non-existent prefix returned: {result.tolist()}")

def main():
    mem_manager, service, hicache, token_size = setup()
    try:
        test_basic_write_read(mem_manager, hicache, token_size)
        test_node_splitting(mem_manager, hicache, token_size)
        test_partial_read(mem_manager, hicache)
    finally:
        service.shutdown()

if __name__ == "__main__":
    main()