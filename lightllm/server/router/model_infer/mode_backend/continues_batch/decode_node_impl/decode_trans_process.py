import rpyc
from typing import List, Dict
from lightllm.utils.log_utils import init_logger
from .decode_infer_rpyc import PDDecodeInferRpcServer
from lightllm.common.mem_manager import MemoryManager
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.net_utils import alloc_can_use_port

logger = init_logger(__name__)


def _init_env(args, nccl_ip, nccl_port, task_in_queue: mp.Queue, task_out_queue: mp.Queue, mem_queues: List[mp.Queue]):
    try:
        # 注册graceful 退出的处理
        from lightllm.utils.graceful_utils import graceful_registry
        import inspect

        graceful_registry(inspect.currentframe().f_code.co_name)
        task_out_queue.put("proc_start")
        mem_managers: List[MemoryManager] = [mem_queue.get(timeout=20) for mem_queue in mem_queues]
        assert len(mem_managers) == args.tp
        task_out_queue.put("get_mem_managers_ok")
        import torch
        import torch.distributed as dist
        from datetime import timedelta

        dist.init_process_group(
            "nccl", init_method=f"tcp://{nccl_ip}:{nccl_port}", rank=1, world_size=2, timeout=timedelta(seconds=60)
        )
        task_out_queue.put("nccl_ok")
        while True:
            move_task: KVMoveTask = task_in_queue.get()
            try:
                for i, mem in enumerate(mem_managers):
                    move_kv_buffer = mem.alloc_kv_move_buffer(len(move_task.key), device=f"cuda:{i}")
                    dist.recv(move_kv_buffer, src=0)
                    mem.kv_buffer[:, move_task.decode_value, :, :] = move_kv_buffer[:, :, :, :]
                torch.cuda.synchronize()
                task_out_queue.put("ok")
            except BaseException as e:
                logger.exception(str(e))
                task_out_queue.put("fail")
                raise e
    except BaseException as e:
        logger.exception(str(e))
        raise e
    return


def start_decode_trans_process(
    args, nccl_ip, nccl_port, task_in_queue: mp.Queue, task_out_queue: mp.Queue, mem_queues: List[mp.Queue]
):
    proc = mp.Process(target=_init_env, args=(args, nccl_ip, nccl_port, task_in_queue, task_out_queue, mem_queues))
    proc.start()
    assert proc.is_alive()
    logger.info(f"decode trans kv process start, nccl_ip: {nccl_ip}, nccl_port: {nccl_port}")
    return proc
