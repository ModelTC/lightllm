import ctypes
import numpy as np
from lightllm.utils.envs_utils import get_unique_server_name
from multiprocessing import shared_memory
from lightllm.utils.log_utils import init_logger
from .req import Req, NormalReq, SplitFuseReq, TokenHealingReq
from .shm_array import ShmArray
from .atomic_array_lock import AtomicShmArrayLock, AtomicLockItem
from .atomic_lock import AtomicShmLock
from .start_args_type import StartArgs
from typing import List
from lightllm.utils.envs_utils import get_env_start_args

logger = init_logger(__name__)


class ShmReqManager:
    def __init__(self):
        self.req_class: Req.__class__ = self.get_req_class_type()
        class_size = ctypes.sizeof(self.req_class)
        self.max_req_num = self.get_max_req_num()
        self.req_shm_byte_size = class_size * self.max_req_num

        self.init_reqs_shm()
        self.init_to_req_objs()
        self.init_to_req_locks()
        self.init_manager_lock()
        self.init_alloc_state_shm()
        return

    def get_req_class_type(self):
        args: StartArgs = get_env_start_args()
        if args.splitfuse_mode:
            return SplitFuseReq
        if args.token_healing_mode:
            return TokenHealingReq
        return NormalReq

    def get_max_req_num(self):
        args: StartArgs = get_env_start_args()
        return args.running_max_req_size

    def init_reqs_shm(self):
        self._init_reqs_shm()

        if self.reqs_shm.size != self.req_shm_byte_size:
            logger.info(f"size not same, unlink lock shm {self.reqs_shm.name} and create again")
            self.reqs_shm.close()
            self.reqs_shm.unlink()
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
        # 用来做为每个进程独立的状态管理，用于申请和
        self.proc_private_get_state = np.zeros(shape=(self.max_req_num,), dtype=np.int32)
        return

    # alloc_req_index 和 release_req_index 是分配资源时使用的接口。
    # 只有管理请求申请和释放的首节点才能调用这个接口。
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

    # get_req_obj_by_index 和 put_back_req_obj 是 分配好后，进行对象获取和
    # 管理的接口，主要是要进行引用计数的管理。
    def get_req_obj_by_index(self, req_index_in_mem):
        assert req_index_in_mem < self.max_req_num
        assert self.proc_private_get_state[req_index_in_mem] == 0
        ans: Req = self.reqs[req_index_in_mem]
        with self.get_req_lock_by_index(req_index_in_mem):
            ans.ref_count = ans.ref_count + 1
        self.proc_private_get_state[req_index_in_mem] = 1
        return ans

    def put_back_req_obj(self, req: Req):
        req_index_in_mem = req.index_in_shm_mem
        assert req_index_in_mem < self.max_req_num
        assert self.proc_private_get_state[req_index_in_mem] == 1
        with self.get_req_lock_by_index(req_index_in_mem):
            req.ref_count = req.ref_count - 1
        self.proc_private_get_state[req_index_in_mem] = 0

    def __del__(self):
        self.reqs = None
