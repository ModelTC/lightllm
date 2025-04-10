import torch
import time
import sys
import inspect
import torch.multiprocessing as mp
from torch.distributed import TCPStore
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask, PDTransJoinInfo, PDTransLeaveInfo, KVMoveTaskGroup
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.distributed.pynccl import PyNcclCommunicator, StatelessP2PProcessGroup

logger = init_logger(__name__)


def _handle_kvmove_task(
    move_tasks: List[KVMoveTask],
    task_out_queue: mp.Queue,
    mem_managers: List[MemoryManager],
    connect_id_to_comm: Dict[str, PyNcclCommunicator],
    connect_id: str,
    dp_size_in_node: int,
):
    total_move_kv_len = sum([task.move_kv_len for task in move_tasks])
    try:
        device_index = connect_id_to_comm[connect_id].device.index
        start = time.time()
        if total_move_kv_len != 0:
            cur_mem = mem_managers[device_index]
            logger.info(f"trans start: {move_tasks[0].to_decode_log_info()}")
            if kv_trans_use_p2p():
                cur_mem.receive_from_prefill_node_p2p(
                    move_tasks, mem_managers, dp_size_in_node, connect_id_to_comm[connect_id]
                )
            else:
                cur_mem.receive_from_prefill_node(
                    move_tasks, mem_managers, dp_size_in_node, connect_id_to_comm[connect_id]
                )
            logger.info(f"trans finished: {move_tasks[0].to_decode_log_info()} move len: {total_move_kv_len}")
        torch.cuda.synchronize()
        logger.info(f"trans cost time: {(time.time() - start)}, {move_tasks[0].to_decode_log_info()}")
        task_out_queue.put("ok")
    except BaseException as e:
        logger.exception(str(e))
        task_out_queue.put("fail")
        raise e


def _handle_prefill_join(
    node_info: PDTransJoinInfo, task_out_queue: mp.Queue, connect_id_to_comm: Dict[str, PyNcclCommunicator]
):
    try:
        store_client = TCPStore(
            host_name=node_info.pd_prefill_nccl_ip, port=node_info.pd_prefill_nccl_port, is_master=False, use_libuv=True
        )
        group = StatelessP2PProcessGroup.create(
            src_id=node_info.prefill_id, dest_id=node_info.decode_id, is_server=False, store=store_client
        )
        comm = PyNcclCommunicator(group, node_info.decode_device_id)
        connect_id_to_comm[node_info.connect_id] = comm
        logger.info(f"{node_info} kv trans connected")
        task_out_queue.put("nccl_ok")
    except Exception as e:
        task_out_queue.put("nccl_fail")
        logger.warning(f"error while connect to prefill node: {e}")


def _init_env(args, device_id: int, task_in_queue: mp.Queue, task_out_queue: mp.Queue, mem_queues: List[mp.Queue]):

    dp_size_in_node = max(1, args.dp // args.nnodes)

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")

        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]

        task_out_queue.put("get_mem_managers_ok")
        connect_id_to_comm: Dict[str, PyNcclCommunicator] = {}
        while True:
            task: Union[KVMoveTaskGroup, PDTransJoinInfo, PDTransLeaveInfo] = task_in_queue.get()
            if isinstance(task, KVMoveTaskGroup):
                _handle_kvmove_task(
                    task.tasks, task_out_queue, mem_managers, connect_id_to_comm, task.connect_id, dp_size_in_node
                )
            elif isinstance(task, PDTransJoinInfo):
                _handle_prefill_join(task, task_out_queue, connect_id_to_comm)
            elif isinstance(task, PDTransLeaveInfo):
                if task.connect_id in connect_id_to_comm:
                    connect_id_to_comm[task.connect_id].destroy()
                    logger.info(f"destory {task} nccl communicator.")
                else:
                    logger.info(f"no connect_id {task.connect_id} found in connect_id_to_comm")

            else:
                logger.warning(f"unexpected task type: {task}")

    except Exception as e:
        logger.error(f"Fatal error happened in kv trans process: {e}")
        raise


def start_decode_trans_process(
    args,
    device_id: int,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    proc = mp.Process(target=_init_env, args=(args, device_id, task_in_queue, task_out_queue, mem_queues))
    proc.start()
    assert proc.is_alive()
    logger.info(f"decode trans kv process for device: {device_id} start!")
    return proc
