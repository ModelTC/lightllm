import rpyc
import sys
import os
import signal
import torch
import time
import threading
from rpyc.utils.classic import obtain
from dataclasses import dataclass
from typing import List, Dict
from rpyc import ThreadedServer
from lightllm.utils.log_utils import init_logger
from .decode_infer_rpyc import PDDecodeInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, UpKVStatus
from lightllm.utils.retry_utils import retry
import queue

logger = init_logger(__name__)


@dataclass
class TransProcessObj:
    prefill_node_id: str = None
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    nccl_ip: str = None
    nccl_port: str = None

    def create(self, prefill_node_id: str, nccl_ip: str, nccl_port: int, manager: "DecodeKVMoveManager"):
        from .decode_trans_process import start_decode_trans_process

        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        proc = start_decode_trans_process(
            manager.args, nccl_ip, nccl_port, task_in_queue, task_out_queue, manager.mem_queues
        )
        assert task_out_queue.get(timeout=30) == "proc_start"
        for obj in manager.infer_rpyc_objs:
            obj.put_mem_manager_to_mem_queue()
        assert task_out_queue.get(timeout=30) == "get_mem_managers_ok"
        assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.prefill_node_id = prefill_node_id
        self.process = proc
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.nccl_ip = nccl_ip
        self.nccl_port = nccl_port
        return

    def __del__(self):
        # 强制关闭连接和杀掉传输进程
        if self.process is not None:
            os.kill(self.process.pid, signal.SIGKILL)
        pass


class DecodeKVMoveManager(rpyc.Service):
    def __init__(self, args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue]):
        super().__init__()
        self.args = args
        self.info_queues = info_queues
        self.mem_queues = mem_queues
        self.infer_rpyc_objs: List[PDDecodeInferRpcServer] = []
        self.node_id_to_trans_obj: Dict[str, TransProcessObj] = {}
        for port in self.args.pd_tp_infer_rpyc_ports:
            con = retry(max_attempts=20, wait_time=2)(rpyc.connect)("localhost", port, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to port: {port} ok")

        # 让推理进程的rpyc server 将 mem manger放入到queue中，下面进行接收
        for obj in self.infer_rpyc_objs:
            obj.put_mem_manager_to_mem_queue()
        self.mem_managers: List[MemoryManager] = []
        for _queue in self.mem_queues:
            self.mem_managers.append(_queue.get())
        logger.info("get mem manager objs from info_queues ok")

        self.task_queue = queue.Queue()

        from .up_status import start_up_kv_status_process

        self.up_status_in_queue = mp.Queue()
        self.up_status_out_queue = mp.Queue()
        start_up_kv_status_process(self.args, self.up_status_in_queue, self.up_status_out_queue)
        self.lock = threading.Lock()
        return

    def exposed_build_trans_process(self, prefill_node_id, nccl_ip, nccl_port):
        prefill_node_id, nccl_ip, nccl_port = list(map(obtain, [prefill_node_id, nccl_ip, nccl_port]))
        if prefill_node_id in self.node_id_to_trans_obj:
            self.node_id_to_trans_obj.pop(prefill_node_id, None)
        tran_obj = TransProcessObj()
        tran_obj.create(prefill_node_id, nccl_ip, nccl_port, self)
        self.node_id_to_trans_obj[prefill_node_id] = tran_obj
        return

    def exposed_request_data_transfer(self, task: KVMoveTask) -> str:
        task = obtain(task)
        # 同时只允许一个任务进行数据迁移操作。
        self.lock.acquire()
        try:
            trans_obj = self.get_trans_obj(task)
            assert trans_obj is not None
            value_list = []
            for conn in self.infer_rpyc_objs:
                decode_value_list = obtain(conn.alloc_to_frozen_some_tokens(task.group_request_id, task.key))
                logger.info(f"wzj {type(decode_value_list)}")
                value_list.append(decode_value_list)

            assert all(isinstance(e, list) for e in value_list)
            task.decode_value = value_list[0]
        except BaseException as e:
            logger.error(str(e))
            self.lock.release()
            return "fail"

        self.task_queue.put(task)
        return "ok"

    def get_trans_obj(self, task: KVMoveTask):
        return self.node_id_to_trans_obj[task.prefill_node_id]

    def handle_loop(self):
        try:
            while True:
                task = self.task_queue.get()
                if not isinstance(task, KVMoveTask):
                    logger.error("not all receive task type is KVMoveTask")
                    sys.exit(-1)

                logger.info("deocode node get task ok")

                try:
                    trans_obj = self.get_trans_obj(task)
                    trans_obj.task_in_queue.put(task, timeout=10)
                    assert trans_obj.task_out_queue.get(timeout=30) == "ok"
                    logger.info("deocode node transfer kv ok")
                    # 成功了将
                    for conn in self.infer_rpyc_objs:
                        conn.put_kv_received_to_radix_cache(task.group_request_id, task.key, task.decode_value)
                    logger.info("decode node put kv to radix cache ok")
                    self.up_status_in_queue.put(UpKVStatus(group_request_id=task.group_request_id))
                    logger.info("decode node up kv status ok")
                except BaseException as e:
                    logger.exception(str(e))
                    # 失败了也需要释放锁定的 token
                    for conn in self.infer_rpyc_objs:
                        conn.fail_to_realese_forzen_tokens(task.group_request_id, task.key, task.decode_value)
                    logger.error("decode kv move task has error, remove the trans_obj")
                    self.node_id_to_trans_obj.pop(task.prefill_node_id, None)
                finally:
                    self.lock.release()
        except BaseException as e:
            logger.exception(str(e))
            raise e


def _init_env(args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue], event: mp.Event):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = DecodeKVMoveManager(args, info_queues, mem_queues)
    t = ThreadedServer(manager, port=args.pd_decode_rpyc_port, protocol_config={"allow_pickle": True})
    threading.Thread(target=lambda: t.start(), daemon=True).start()

    event.set()

    # 进入主循环
    manager.handle_loop()
    return


def start_decode_kv_move_manager_process(args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queues, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move process started")
    return
