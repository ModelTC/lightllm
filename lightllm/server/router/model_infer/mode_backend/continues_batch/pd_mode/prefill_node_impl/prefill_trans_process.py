import torch
import time
import sys
import inspect
import torch.multiprocessing as mp
from torch.distributed import TCPStore
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.mem_manager import MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask, PDTransJoinInfo, PDTransLeaveInfo
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.distributed.pynccl import StatelessP2PProcessGroup, PyNcclCommunicator


logger = init_logger(__name__)

def _handle_kvmove_task(move_tasks: List[KVMoveTask], task_out_queue: mp.Queue,
                        mem_managers: List[MemoryManager], decode_to_comm: Dict[int, PyNcclCommunicator],
                        dp_size_in_node: int):
    total_move_kv_len = sum([task.move_kv_len for task in move_tasks])
    try:
        decode_id = move_tasks[0].decode_node.node_id
        device_index = decode_to_comm[decode_id].device.index
        torch.cuda.set_device(device_index)
        start = time.time()
        if total_move_kv_len != 0:
            logger.info(f"trans start: {move_tasks[0].to_prefill_log_info()}")
            cur_mem = mem_managers[device_index]
            if kv_trans_use_p2p():
                cur_mem.send_to_decode_node_p2p(move_tasks, mem_managers, dp_size_in_node, decode_to_comm[decode_id])
            else:
                cur_mem.send_to_decode_node(move_tasks, mem_managers, dp_size_in_node, decode_to_comm[decode_id])
            logger.info(f"trans finished: {move_tasks[0].to_prefill_log_info()} move len: {total_move_kv_len}")
        torch.cuda.synchronize()
        logger.info(
            f"trans cost time: {(time.time() - start)},"
            f"move_total_kv_len: {total_move_kv_len}, {move_tasks[0].to_prefill_log_info()}"
        )
        task_out_queue.put("ok")
    except BaseException as e:
        logger.exception(str(e))
        task_out_queue.put("fail")

def _handle_decode_join(node_info: PDTransJoinInfo, task_out_queue: mp.Queue, decode_to_comm: Dict[str, PyNcclCommunicator], store: TCPStore):
    try:
        group = StatelessP2PProcessGroup.create(node_info.prefill_id, node_info.decode_id, True, store)
        comm = PyNcclCommunicator(group, node_info.prefill_device_id)
        decode_to_comm[node_info.decode_id] = comm
        logger.info(f"{node_info} kv trans connected!")
        task_out_queue.put("nccl_ok")
    except Exception as e:
        logger.warning(f"error while connect to decode node: {e}")

def _init_env(
    args,
    store_ip,
    store_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],):
    try:
        graceful_registry(inspect.currentframe().f_code.co_name)
        master_store = TCPStore(host_name=store_ip, port=store_port, is_master=True, use_libuv=True)
        dp_size_in_node = max(1, args.dp // args.nnodes)
        node_world_size = args.tp // args.nnodes
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=60) for mem_queue in mem_queues]
        assert len(mem_managers) == node_world_size
        task_out_queue.put("get_mem_managers_ok")
        decode_to_comm: Dict[int, PyNcclCommunicator] = {}

        while True:
            task: Union[List, PDTransJoinInfo, PDTransLeaveInfo] = task_in_queue.get()
            if isinstance(task, List):
                _handle_kvmove_task(task, task_out_queue, mem_managers, decode_to_comm, dp_size_in_node)
            elif isinstance(task, PDTransJoinInfo):
                _handle_decode_join(task, task_out_queue, decode_to_comm, master_store)
            elif isinstance(task, PDTransLeaveInfo):
                decode_to_comm[task.decode_id].destroy()
                logger.info(f"destory {task.decode_id} nccl communicator.")
            else:
                logger.warning(f'unexpected task type: {task}')

    except Exception as e:
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


def start_decode_trans_process(
    args,
    store_ip,
    store_port,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
    mem_queues: List[mp.Queue],
):
    proc = mp.Process(
        target=_init_env, args=(args, store_ip, store_port, task_in_queue, task_out_queue, mem_queues)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"trans kv process started!")
    return proc