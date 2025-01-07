import ctypes
import numpy as np
from lightllm.utils.envs_utils import get_unique_server_name
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from .req import Req, NormalReq
from .shm_array import ShmArray
from .atomic_array_lock import AtomicShmArrayLock, AtomicLockItem
from .atomic_lock import AtomicShmLock
from typing import List

logger = init_logger(__name__)


class ShmReqManager:
    def __init__(self, req_class: Req.__class__, max_req_num: int):
        class_size = ctypes.sizeof(req_class)
        self.req_class = req_class
        self.req_shm_byte_size = class_size * max_req_num
        self.max_req_num = max_req_num
        self.init_reqs_shm()
        self.init_to_req_objs()
        self.init_to_req_locks()
        self.init_manager_lock()
        self.init_alloc_state_shm()
        return

    def init_reqs_shm(self):
        self._init_reqs_shm()

        if self.reqs_shm.size != self.req_shm_byte_size:
            logger.info(f"size not same, unlink lock shm {self.reqs_shm.name} and create again")
            self.reqs_shm.unlink()
            self.reqs_shm.close()
            self.reqs_shm = None
            self._init_reqs_shm()

    def _init_reqs_shm(self):
        shm_name = f"{get_unique_server_name()}_req_shm_total"
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.req_shm_byte_size)
            logger.info(f"create lock shm {shm_name}")
        except:
            shm = shared_memory.SharedMemory(name=shm_name, create=False, size=self.req_shm_byte_size)
            logger.info(f"link lock shm {shm_name}")
        self.reqs_shm = shm
        return

    def init_to_req_objs(self):
        self.reqs: List[Req] = (self.req_class * self.max_req_num).from_buffer(self.reqs_shm.buf)
        for i in range(self.max_req_num):
            self.reqs[i].ref_count = 0
            self.reqs[i].index_in_shm_mem = i
        return

    def init_to_req_locks(self):
        array_lock_name = f"{get_unique_server_name()}_array_reqs_lock"
        self.reqs_lock = AtomicShmArrayLock(array_lock_name, self.max_req_num)
        return

    def get_req_lock_by_index(self, req_index_in_mem: int) -> AtomicLockItem:
        return self.reqs_lock.get_lock_context(req_index_in_mem)

    def init_manager_lock(self):
        lock_name = f"{get_unique_server_name()}_shm_reqs_manager_lock"
        self.manager_lock = AtomicShmLock(lock_name)
        return

    def init_alloc_state_shm(self):
        shm_name = f"{get_unique_server_name()}_req_alloc_states"
        self.alloc_state_shm = ShmArray(shm_name, (self.max_req_num,), np.int32)
        self.alloc_state_shm.create_shm()
        self.alloc_state_shm.arr[:] = 0
        return

    def alloc_req_index(self):
        with self.manager_lock:
            indices = np.where(self.alloc_state_shm.arr == 0)[0]
            if len(indices) == 0:
                return None
            else:
                ans = indices[0]
                self.alloc_state_shm.arr[ans] = 1
                return ans

    def release_req_index(self, req_index_in_mem):
        assert req_index_in_mem < self.max_req_num
        with self.manager_lock:
            assert self.alloc_state_shm.arr[req_index_in_mem] == 1
            self.alloc_state_shm.arr[req_index_in_mem] = 0
        return
