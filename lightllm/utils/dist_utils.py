import torch.distributed as dist
import os
import torch


def set_environ(environ_name, value):
    os.environ[environ_name] = str(value)


def get_environ(environ_name):
    value = os.getenv(environ_name, None)
    if value is None:
        raise RuntimeError(f"{environ_name} is not set")
    return value


def _init_distributed_env(kvargs):
    set_global_rank(kvargs["rank_id"])  # TODO: implement DP
    set_global_world_size(kvargs["world_size"])
    set_dp_world_size(kvargs.get("dp_size", 1))
    # set_current_dp_rank()
    set_current_tp_rank(kvargs["rank_id"])
    size_per_node = (kvargs["world_size"] + kvargs["args"].nnodes - 1) // kvargs["args"].nnodes
    local_tp_rank = kvargs["rank_id"] - size_per_node * kvargs["args"].node_rank
    set_current_device_id(local_tp_rank)
    torch.cuda.set_device(local_tp_rank)
    if kvargs["world_size"] > 1:
        dist.init_process_group(
            "nccl",
            init_method=f'tcp://{kvargs["nccl_host"]}:{kvargs["nccl_port"]}',
            rank=kvargs["rank_id"],
            world_size=kvargs["world_size"],
        )
        # warmup nccl communicator
        _a = torch.zeros([1]).to(f"cuda:{local_tp_rank}")
        dist.all_reduce(_a)
        del _a


def set_global_rank(global_rank: int):
    set_environ("GLOBAL_RANK", global_rank)


def get_global_rank():
    return int(get_environ("GLOBAL_RANK"))


def set_global_world_size(world_size: int):
    set_environ("GLOBAL_WORLD_SIZE", world_size)


def get_global_world_size():
    return int(get_environ("GLOBAL_WORLD_SIZE"))


def set_dp_world_size(world_size: int):
    set_environ("DP_WORLD_SIZE", world_size)


def get_dp_world_size():
    return int(get_environ("DP_WORLD_SIZE"))


def set_current_dp_rank(rank: int):
    set_environ("CURRENT_DP_RANK", rank)


def get_current_dp_rank():
    assert False, "not implemented"
    return int(get_environ("CURRENT_DP_RANK"))


def set_current_tp_rank(rank: int):
    set_environ("CURRENT_TP_RANK", rank)


def set_current_device_id(device_id: int):
    set_environ("CURRENT_DEVICE_ID", device_id)


def get_current_device_id():
    return int(get_environ("CURRENT_DEVICE_ID"))
