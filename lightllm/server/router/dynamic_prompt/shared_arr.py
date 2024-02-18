import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory 

class SharedArray:
    def __init__(self, name, shape, dtype):
        try:
            print(f"create shm {name}")
            shm = shared_memory.SharedMemory(name=name, create=True, size=np.cumprod(shape) * dtype.itemsize)
        except:
            print(f"link shm {name}")
            shm = shared_memory.SharedMemory(name=name, create=False, size=np.cumprod(shape) * dtype.itemsize)
        self.arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)


class SharedIdxNode:
    def __init__(self, manager, idx) -> None:
        self.manager = manager
        self.idx = idx

    def get_idx(self):
        return self.idx
    
    def get_parent_idx(self):
        return self.manager._values[self.idx, 3]
    
    def set_parent_idx(self, p_idx):
        self.manager._values[self.idx, 3] = p_idx
    
    def get_parent_idx_shared_node(self):
        return SharedIdxNode(self.manager, self.manager._values[self.idx, 3])
    
    def get_node_value_len(self):
        return self.manager._values[self.idx, 4]
    
    def set_node_value_len(self, value_len):
        self.manager._values[self.idx, 4] = value_len
    
    def get_node_prefix_total_len(self):
        return self.manager._values[self.idx, 5]
    
    def set_node_prefix_total_len(self, prefix_total_len):
        self.manager._values[self.idx, 5] = prefix_total_len


class SharedTreeIdxManager:
    VALUE_INDEX = 0
    PRE_INDEX = 1
    NEXT_INDEX = 2

    def __init__(self, total_token_num, tp_id) -> None:
        self.size = total_token_num + 2 # 因为 0 号节点不分配，所以为了满足充分可用性需要 + 2.
        # 第二维对应信息 0 idx 1 pre index 2 next index 用于链表管理  3 tree_node parent node idx 4 tree_node value len 5 tree node prefix total len 
        self._values = SharedArray(f"SharedTreeIdx_{tp_id}", shape=(self.size, 6), dtype=np.int64)
        # idx
        self._values[:, self.VALUE_INDEX] = np.arange(0, self.size, 1)
        # pre     
        self._values[0,  self.PRE_INDEX] = -1
        self._values[1:, self.PRE_INDEX] = np.arange(0, self.size - 1, 1)
        # next
        self._values[0:self.size - 1, self.NEXT_INDEX] = np.arange(1, self.size, 1)
        self._values[self.size - 1, self.NEXT_INDEX] = -1

        # tree node value
        self._values[:, 3] = -1
        self._values[:, 4] = 0
        self._values[:, 5] = 0
    
    def alloc(self):
        if self._values[0, self.NEXT_INDEX] != -1:
            alloc_idx = self._values[0, self.NEXT_INDEX]
            if self._values[alloc_idx, self.NEXT_INDEX] == -1:
                self._values[0, self.NEXT_INDEX] = -1
                ans = SharedIdxNode(self, alloc_idx)
            
            nn_idx = self._values[alloc_idx, self.NEXT_INDEX]
            self._values[0, self.NEXT_INDEX] = nn_idx
            self._values[nn_idx, self.PRE_INDEX] = 0
            ans = SharedIdxNode(self, alloc_idx)
            # 初始化值
            ans.set_parent_idx(-1)
            ans.set_node_value_len(0)
            ans.set_node_prefix_total_len(0)
            return ans 
        
        assert False, "error cannot alloc"

    def free(self, idx):
        nn_idx = self._values[0, self.NEXT_INDEX]
        self._values[0, self.NEXT_INDEX] = idx
        self._values[idx, self.PRE_INDEX] = 0
        self._values[idx, self.NEXT_INDEX] = nn_idx
        if nn_idx != -1:
            self._values[nn_idx, self.PRE_INDEX] = idx
        return
    
    def get_shared_node(self, idx):
        return SharedIdxNode(self, idx)


