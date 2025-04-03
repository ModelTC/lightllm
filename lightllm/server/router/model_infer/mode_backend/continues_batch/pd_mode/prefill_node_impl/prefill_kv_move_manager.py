import asyncio
import time
import rpyc
import sys
import os
import gc
import signal
import copy
import numpy as np
import psutil
import threading
import inspect
import collections
from dataclasses import dataclass
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
from .prefill_infer_rpyc import PDPrefillInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask, PDTransJoinInfo, PDTransLeaveInfo
from lightllm.utils.net_utils import find_available_port
from lightllm.utils.retry_utils import retry
from rpyc.utils.classic import obtain
from rpyc import AsyncResult
from lightllm.utils.net_utils import get_hostname_ip
from ..task_queue import TaskQueue
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name

KV_MOVE_MAX_NUM = 16
KV_MOVE_MAX_START_CNT = 3

logger = init_logger(__name__)


@dataclass
class TransProcessObj:
    decode_node_id: int = None
    rpyc_conn: object = None  # rpyc_con 的连接对象
    process: mp.Process = None
    task_in_queue: mp.Queue = None
    task_out_queue: mp.Queue = None
    device_index: int = None  # 使用的gpu序号
    manager: "PrefillKVMoveManager" = None
    has_error: bool = False
    request_kv_trans_task_queue: TaskQueue = None
    request_thread: threading.Thread = None
    ready_kv_trans_task_queue: TaskQueue = None
    kv_trans_thread: threading.Thread = None
    latest_check_time: float = None

    def create(
        self, decode_node_id: int, decode_node_ip: str, decode_node_rpyc_port: int, manager: "PrefillKVMoveManager"
    ):
        con = rpyc.connect(
            host=decode_node_ip, port=decode_node_rpyc_port, config={"allow_pickle": True}, keepalive=True
        )

        device_index = manager.get_next_device_index()  # 分配 trans 进程使用的显卡
        prefill_node_id = manager.args.pd_node_id
        task_in_queue = manager.kv_trans_task_in_queues[device_index]
        task_out_queue = manager.kv_trans_task_out_queues[device_index]

        with manager.device_locks[device_index]:
            task_in_queue.put(
                PDTransJoinInfo(
                    prefill_id=prefill_node_id,
                    prefill_device_id=device_index,
                    pd_prefill_nccl_ip=manager.host_ip,
                    pd_prefill_nccl_port=manager.kv_trans_ports[device_index],
                    decode_id=decode_node_id,
                    decode_device_id=-1,
                )
            )

            # 异步调用, 让decode节点建立与prefill节点进行nccl通信的进程
            max_kv_trans_token_num = obtain(
                con.root.build_trans_process(
                    prefill_node_id,
                    manager.host_ip,
                    manager.kv_trans_ports[device_index],
                    manager.args.max_total_token_num,
                )
            )
            self.max_kv_trans_token_num = max_kv_trans_token_num
            assert task_out_queue.get(timeout=60) == "nccl_ok"

        self.decode_node_id = decode_node_id
        self.prefill_node_id = prefill_node_id
        self.rpyc_conn = con
        self.task_in_queue = task_in_queue
        self.task_out_queue = task_out_queue
        self.device_index = device_index
        self.manager = manager
        self.latest_check_time = time.time()
        self.process = manager.kv_trans_processes[device_index]

        self.request_kv_trans_task_queue = TaskQueue(
            get_func=self._get_request_tasks, fail_func=self.manager.put_to_release_task_queue
        )
        self.request_thread = threading.Thread(target=self.request_kv_trans_loop, daemon=True)
        self.request_thread.start()

        self.ready_kv_trans_task_queue = TaskQueue(lambda datas: datas[0:1], self.manager.put_to_release_task_queue)
        self.kv_trans_thread = threading.Thread(target=self.kv_trans_handle_loop, daemon=True)
        self.kv_trans_thread.start()
        return

    def _get_request_tasks(self, datas: List[KVMoveTask]):
        ans_list = []
        token_num = 0
        for task in datas:
            if token_num + len(task.prefill_token_indexes) <= self.max_kv_trans_token_num:
                ans_list.append(task)
                token_num += len(task.prefill_token_indexes)
            else:
                break
        return ans_list

    def check_connect(self, raise_exception=True):
        try:
            self.rpyc_conn.root.check_alive()
        except BaseException as e:
            self.set_has_error()
            if raise_exception:
                raise e
        return

    def check_trans_process(self, raise_exception=True):
        process = psutil.Process(self.process.pid)
        if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
            self.set_has_error()
            if raise_exception:
                raise Exception(f"trans process: {self.process.pid} is dead")
        return

    def timer_check_status(self, raise_exception=True):
        if time.time() - self.latest_check_time >= 2.0:
            self.latest_check_time = time.time()
            self.check_trans_process(raise_exception=raise_exception)
            self.check_connect(raise_exception=raise_exception)
            if self.has_error:
                self.manager.remove_trans_obj(self.decode_node_id)
        return

    def request_kv_trans_loop(self):
        func_name = self.request_kv_trans_loop.__name__

        while not self.has_error:
            move_tasks: List[KVMoveTask] = self.request_kv_trans_task_queue.get_tasks(
                log_tag="request_kv_trans_task_queue"
            )
            if len(move_tasks) == 0:
                # 周期检查通信状态
                self.timer_check_status(raise_exception=False)
                time.sleep(0.01)
                continue
            try:
                self.timer_check_status(raise_exception=True)
                for move_task in move_tasks:
                    logger.info(
                        f"{func_name} get task {move_task.to_prefill_log_info()} "
                        f"queue time {move_task.get_cost_time()} s "
                    )

                trans_move_tasks = [copy.copy(move_task) for move_task in move_tasks]
                for trans_move_task in trans_move_tasks:
                    trans_move_task.prefill_token_indexes = None

                mark_start = time.time()
                move_kv_lens = self.rpyc_conn.root.request_data_transfer(trans_move_tasks)
                move_kv_lens = obtain(move_kv_lens)
                request_data_transfer_cost_time = time.time() - mark_start

                logger.info(
                    f"{func_name} request_data_transfer ok, {move_tasks[0].to_prefill_log_info()}"
                    f" cost time: {request_data_transfer_cost_time} s"
                )

                ok_trans_list = []
                for i, move_task in enumerate(move_tasks.copy()):
                    if move_kv_lens[i] is not None:
                        move_task.move_kv_len = move_kv_lens[i]
                        ok_trans_list.append(move_task)
                        move_tasks.remove(move_task)
                    else:
                        logger.info(f"prefill node kv move task req_id: {move_task.id()} not send, decode is busy")

                if len(ok_trans_list) != 0:
                    self.ready_kv_trans_task_queue.put(ok_trans_list)

            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.manager.remove_trans_obj(self.decode_node_id)
                self.request_kv_trans_task_queue.clear_tasks()

            finally:
                self.manager.put_to_release_task_queue(move_tasks)

        logger.error(f"{func_name}, decode id {self.decode_node_id} device_index {self.device_index} thread quit")
        return

    def _transfer_kv(self, move_tasks: List[KVMoveTask]):
        with self.manager.device_locks[self.device_index]:
            self.task_in_queue.put(move_tasks.copy(), timeout=10)
            assert self.task_out_queue.get(timeout=60) == "ok"
            self.manager.put_to_release_task_queue(move_tasks)

            logger.info(
                f"_transfer_kv data ok, req_id: {move_tasks[0].id()}"
                f" cost total time: {move_tasks[0].get_cost_time()} s"
            )
            move_tasks.clear()

    def kv_trans_handle_loop(self):
        func_name = self.kv_trans_handle_loop.__name__
        while not self.has_error:
            move_tasks: List[List[KVMoveTask]] = self.ready_kv_trans_task_queue.get_tasks(
                log_tag="ready_kv_trans_task_queue"
            )
            if len(move_tasks) == 0:
                self.timer_check_status(raise_exception=False)
                time.sleep(0.01)
                continue

            if len(move_tasks) != 1:
                logger.error(f"error get kv trans move_tasks, must be 1, get {len(move_tasks)}")
                assert len(move_tasks) == 1

            move_tasks = move_tasks[0]

            try:
                self.timer_check_status(raise_exception=True)
                for move_task in move_tasks:
                    logger.info(
                        f"{func_name} get task {move_task.to_prefill_log_info()} to start kv move"
                        f"queue time {move_task.get_cost_time()} s "
                    )

                if not kv_trans_use_p2p():
                    with self.manager.kv_trans_lock:
                        self._transfer_kv(move_tasks)
                else:
                    self._transfer_kv(move_tasks)
            except BaseException as e:
                logger.exception(str(e))
                self.set_has_error()
                self.manager.remove_trans_obj(self.decode_node_id)
                self.ready_kv_trans_task_queue.clear_tasks()
            finally:
                self.manager.put_to_release_task_queue(move_tasks)

        logger.error(f"trans kv thread, decode id {self.decode_node_id} device_index {self.device_index} thread quit")
        return

    def wait_thread_quit(self):
        if self.request_thread is not None:
            if self.request_thread.is_alive():
                try:
                    self.request_thread.join()
                except:
                    pass
        if self.kv_trans_thread is not None:
            if self.kv_trans_thread.is_alive():
                try:
                    self.kv_trans_thread.join()
                except:
                    pass
        return

    def has_error_status(self):
        try:
            assert self.has_error is False
            assert self.request_thread.is_alive()
            assert self.kv_trans_thread.is_alive()
        except BaseException as e:
            logger.exception(str(e))
            self.set_has_error()
            return True

        return False

    def set_has_error(self):
        self.has_error = True
        try:
            self.request_kv_trans_task_queue.has_error = True
            self.ready_kv_trans_task_queue.has_error = True
        except:
            pass
        return

    def __del__(self):
        logger.error(f"trans obj del start, decode node id {self.decode_node_id} device_index {self.device_index}")

        try:
            self.set_has_error()
            self.wait_thread_quit()
            self.task_in_queue.put(PDTransLeaveInfo(decode_id=self.decode_node_id, prefill_id=self.prefill_node_id))
            if self.request_kv_trans_task_queue is not None:
                self.request_kv_trans_task_queue.clear_tasks()
            if self.ready_kv_trans_task_queue is not None:
                self.ready_kv_trans_task_queue.clear_tasks()
        except BaseException as e:
            logger.exception(str(e))

        logger.error(f"trans obj deled, decode node id {self.decode_node_id} device_index {self.device_index}")


class PrefillKVMoveManager:
    def __init__(self, args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.mem_queues = mem_queues
        self.infer_rpyc_objs: List[PDPrefillInferRpcServer] = []
        self.node_id_to_trans_obj: Dict[str, TransProcessObj] = {}
        for port in self.args.pd_node_infer_rpyc_ports:
            socket_path = f"/tmp/{get_unique_server_name()}_prefill_node_infer_rpyc_{port}"
            from rpyc.utils.factory import unix_connect

            con = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to infer rpyc port: {port} ok")
        self.host_ip = get_hostname_ip()
        if self.host_ip is None:
            self.host_ip = args.host

        self.infer_rpyc_lock = threading.Lock()

        self.kv_trans_lock = threading.Lock()
        # 需要每个卡有一个锁来规划每次只能有一个tran obj 操作对应显卡上的传输任务。
        self.device_locks = [threading.Lock() for _ in range(self.node_world_size)]

        # 释放token的task队列
        self.release_task_queue = TaskQueue(lambda datas: datas[0:KV_MOVE_MAX_NUM], fail_func=None)
        self.release_tasks_thread = threading.Thread(target=self.handle_release_task_loop, daemon=True)
        self.release_tasks_thread.start()

        # start a single kv trans process

        from .prefill_trans_process import start_prefill_trans_process

        self.kv_trans_ports = [None] * self.node_world_size
        self.kv_trans_processes = [None] * self.node_world_size
        self.kv_trans_task_in_queues = [None] * self.node_world_size
        self.kv_trans_task_out_queues = [None] * self.node_world_size
        self.kv_trans_process_start_cnt = [0] * self.node_world_size

        for device_id in range(self.node_world_size):
            assert self.start_trans_process(device_id)

        return

    def put_to_release_task_queue(self, task: Union[KVMoveTask, List[KVMoveTask]]):
        if isinstance(task, KVMoveTask):
            self.release_task_queue.put(task)
        elif isinstance(task, list):
            self.release_task_queue.put_list(task)
        else:
            logger.error("error input in put_to_release_task_queue func")
        return

    def handle_release_task_loop(self):
        while True:
            handle_list: List[KVMoveTask] = self.release_task_queue.get_tasks(log_tag="release_task_queue")
            if len(handle_list) == 0:
                time.sleep(0.01)
            else:
                self._remove_req_refs_from_prompt_cache(handle_list)
        return

    def start_trans_process(self, device_id: int):
        task_in_queue = mp.Queue()
        task_out_queue = mp.Queue()
        kv_trans_port = find_available_port(self.args.pd_p_allowed_port_min, self.args.pd_p_allowed_port_max)
        self.kv_trans_process_start_cnt[device_id] += 1

        if self.kv_trans_processes[device_id]:
            # force kill
            try:
                self.remove_trans_obj_by_deviceid(device_id)
                process = psutil.Process(self.kv_trans_processes[device_id].pid)
                process.kill()
                self.kv_trans_processes[device_id] = None
            except Exception:
                pass

        try:
            from .prefill_trans_process import start_prefill_trans_process

            kv_trans_process = start_prefill_trans_process(
                self.args,
                self.host_ip,
                kv_trans_port,
                device_id,
                task_in_queue,
                task_out_queue,
                self.mem_queues,
            )
            assert task_out_queue.get(timeout=30) == "proc_start"
            self._put_mem_manager_to_mem_queue()
            assert task_out_queue.get(timeout=60) == "get_mem_managers_ok"

            self.kv_trans_processes[device_id] = kv_trans_process
            self.kv_trans_task_in_queues[device_id] = task_in_queue
            self.kv_trans_task_out_queues[device_id] = task_out_queue
            self.kv_trans_ports[device_id] = kv_trans_port

            return True
        except Exception as e:
            logger.warning(f"Failed start kv trans process for device {device_id}: {e}")
            return False

    def check_trans_process(self, raise_exception=True):
        at_least_one_alive = False
        for device_id in range(self.node_world_size):
            if not self.is_kv_trans_process_alive(device_id):
                continue

            process = psutil.Process(self.kv_trans_processes[device_id].pid)
            if not (process.is_running() and process.status() != psutil.STATUS_ZOMBIE):
                logger.error(f"kv trans process for device: {device_id} dead!!!, try start again...")
                self.start_trans_process(device_id)
            else:
                at_least_one_alive = True

        if not at_least_one_alive:
            if raise_exception:
                raise Exception("All trans process are dead!!!")

        return

    def check_trans_process_loop(self):
        try:
            while True:
                self.check_trans_process()
                time.sleep(10.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            # kill parent process if any exception occurred
            os.kill(os.getppid(), signal.SIGTERM)
            raise e

    def is_kv_trans_process_alive(self, device_id):
        return self.kv_trans_process_start_cnt[device_id] <= KV_MOVE_MAX_START_CNT

    def get_next_device_index(self):

        counts = [
            0 if self.is_kv_trans_process_alive(device_id) else (1 << 20) for device_id in range(self.node_world_size)
        ]
        for obj in self.node_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        device_index = int(np.argmin(counts))
        return device_index

    def get_trans_obj(self, task: KVMoveTask):
        self.remove_dead_trans_obj()
        if task.decode_node.node_id not in self.node_id_to_trans_obj:
            gc.collect()
            trans_obj = TransProcessObj()
            trans_obj.create(task.decode_node.node_id, task.decode_node.ip, task.decode_node.rpyc_port, self)
            self.node_id_to_trans_obj[task.decode_node.node_id] = trans_obj
        return self.node_id_to_trans_obj[task.decode_node.node_id]

    def remove_trans_obj(self, decode_node_id):
        if decode_node_id in self.node_id_to_trans_obj:
            trans_obj = self.node_id_to_trans_obj.pop(decode_node_id, None)
            if trans_obj is not None:
                trans_obj.set_has_error()
                logger.error(f"remove tran obj id {trans_obj.decode_node_id}")
        return

    def remove_dead_trans_obj(self):
        del_node_ids = []
        for node_id, t_obj in self.node_id_to_trans_obj.items():
            if t_obj.has_error_status():
                del_node_ids.append(node_id)

        for node_id in del_node_ids:
            self.node_id_to_trans_obj.pop(node_id, None)

        if len(del_node_ids) != 0:
            gc.collect()
        return

    def remove_trans_obj_by_deviceid(self, device_id):
        for node_id, t_obj in self.node_id_to_trans_obj.items():
            if t_obj.device_index == device_id:
                self.remove_dead_trans_obj(node_id)

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                move_task: KVMoveTask = self.info_queue.get()
                try:
                    trans_obj = self.get_trans_obj(move_task)
                    trans_obj.request_kv_trans_task_queue.put(move_task)
                except BaseException as e:
                    logger.exception(str(e))
                    self.put_to_release_task_queue(move_task)
                finally:
                    trans_obj = None

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    def _remove_req_refs_from_prompt_cache(self, tasks: List[KVMoveTask]):
        with self.infer_rpyc_lock:
            dp_to_tasks = collections.defaultdict(list)
            for task in tasks:
                dp_to_tasks[task.prefill_dp_index].append(task)
            futures: List[AsyncResult] = []
            for prefill_dp_index, _tasks in dp_to_tasks.items():
                conn_start = prefill_dp_index * self.dp_world_size
                conn_end = (prefill_dp_index + 1) * self.dp_world_size
                conns = self.infer_rpyc_objs[conn_start:conn_end]
                for conn in conns:
                    futures.append(
                        rpyc.async_(conn.remove_req_refs_from_prompt_cache)([task.group_request_id for task in _tasks])
                    )
            asyncio.run(self.wait_all_future_finish(futures))
        return

    def _put_mem_manager_to_mem_queue(self):
        with self.infer_rpyc_lock:
            for obj in self.infer_rpyc_objs:
                obj.put_mem_manager_to_mem_queue()
        return

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return


def _init_env(args, info_queue: mp.Queue, mem_queues: List[mp.Queue], event: mp.Event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)

    manager = PrefillKVMoveManager(args, info_queue, mem_queues)
    kv_trans_process_check = threading.Thread(target=manager.check_trans_process_loop, daemon=True)
    kv_trans_process_check.start()
    event.set()
    # 进入主循环
    manager.task_dispatcher_loop()
    return


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue, mem_queues: List[mp.Queue]):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, mem_queues, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return
