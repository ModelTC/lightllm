import rpyc
import sys
import os
import signal
import torch
import time
import threading
from dataclasses import dataclass
from typing import List, Dict
from lightllm.utils.log_utils import init_logger
from .prefill_infer_rpyc import PDPrefillInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.net_utils import alloc_can_use_port
from lightllm.utils.retry_utils import retry
from rpyc.utils.classic import obtain

logger = init_logger(__name__)


@dataclass
class TransProcessObj:
    decode_node_id: str = None
    rpyc_conn: object = None  # rpyc_con 的连接对象
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    nccl_ip: str = None
    nccl_port: str = None

    def create(
        self, decode_node_id: str, decode_node_ip: str, decode_node_rpyc_port: int, manager: "PrefillKVMoveManager"
    ):
        con = rpyc.connect(
            host=decode_node_ip, port=decode_node_rpyc_port, config={"allow_pickle": True}, keepalive=True
        )
        nccl_ip = manager.args.host
        nccl_port = manager.get_next_nccl_port()
        from .prefill_trans_process import start_prefill_trans_process

        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        proc = start_prefill_trans_process(
            manager.args, nccl_ip, nccl_port, task_in_queue, task_out_queue, manager.mem_queues
        )
        assert task_out_queue.get(timeout=30) == "proc_start"
        for obj in manager.infer_rpyc_objs:
            obj.put_mem_manager_to_mem_queue()
        assert task_out_queue.get(timeout=30) == "get_mem_managers_ok"
        prefill_node_id = manager.args.pd_node_id
        con.root.build_trans_process(prefill_node_id, nccl_ip, nccl_port)  # 异步调用, 让decode节点建立与prefill节点进行nccl通信的进程
        assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.decode_node_id = decode_node_id
        self.rpyc_conn = con
        self.process = proc
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.nccl_port = nccl_port
        self.nccl_ip = nccl_ip
        return

    def __del__(self):
        # 强制关闭连接和杀掉传输进程
        if self.process is not None:
            os.kill(self.process.pid, signal.SIGKILL)
        pass


class PrefillKVMoveManager:
    def __init__(self, args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue]):
        self.args = args
        self.info_queues = info_queues
        self.mem_queues = mem_queues
        self.infer_rpyc_objs: List[PDPrefillInferRpcServer] = []
        self.node_id_to_trans_obj: Dict[str, TransProcessObj] = {}
        self.kv_move_used_ports = alloc_can_use_port(self.args.pd_p_allowed_port_min, self.args.pd_p_allowed_port_max)
        self.port_alloc_index = 0
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

    def get_next_nccl_port(
        self,
    ):
        port = self.kv_move_used_ports[self.port_alloc_index % (len(self.kv_move_used_ports))]
        self.port_alloc_index += 1
        return port

    def get_trans_obj(self, task: KVMoveTask):
        if task.decode_node.node_id not in self.node_id_to_trans_obj:
            trans_obj = TransProcessObj()
            trans_obj.create(task.decode_node.node_id, task.decode_node.ip, task.decode_node.rpyc_port, self)
            self.node_id_to_trans_obj[task.decode_node.node_id] = trans_obj
        return self.node_id_to_trans_obj[task.decode_node.node_id]

    def handle_loop(self):
        try:
            while True:
                move_tasks: List[KVMoveTask] = []

                # 4 个推理子进程会执行相同的代码
                for _queue in self.info_queues:
                    move_task: KVMoveTask = _queue.get()
                    move_tasks.append(move_task)

                if not all(isinstance(e, KVMoveTask) for e in move_tasks):
                    logger.error("not all receive task type is PrefillKVMoveTask")
                    sys.exit(-1)

                logger.info("prefill node get task ok")

                move_task = move_tasks[0]
                try:
                    trans_obj = self.get_trans_obj(move_task)
                    # 申请传输
                    ans = obtain(trans_obj.rpyc_conn.root.request_data_transfer(move_task))
                    if ans != "ok":
                        raise Exception("not ok")
                    logger.info("prefill node request_data_transfer ok")

                    # 开始传输直到完成
                    trans_obj.task_in_queue.put(move_task, timeout=10)
                    assert trans_obj.task_out_queue.get(timeout=30) == "ok"
                    logger.info("prefill node transfer data ok")

                except BaseException as e:
                    logger.exception(str(e))
                    logger.error("kv move task has error, remove the trans_obj")
                    self.node_id_to_trans_obj.pop(move_task.decode_node.node_id, None)

                finally:
                    # 解除对prefill token的占用状态。
                    for infer_rpyc in self.infer_rpyc_objs:
                        infer_rpyc.remove_req_refs_from_prompt_cache(
                            move_task.group_request_id, move_task.key, move_task.prefill_value
                        )
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e


def _init_env(args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue], event: mp.Event):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = PrefillKVMoveManager(args, info_queues, mem_queues)
    event.set()
    # 进入主循环
    manager.handle_loop()
    return


def start_prefill_kv_move_manager_process(args, info_queues: List[mp.Queue], mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queues, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move process started")
    return
