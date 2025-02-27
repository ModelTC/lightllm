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
    set_global_rank(kvargs["rank_id"])
    set_global_world_size(kvargs["world_size"])
    set_dp_size(kvargs.get("dp_size", 1))
    set_dp_world_size(get_global_world_size() // get_dp_size())
    set_current_dp_rank(get_global_rank() // get_dp_world_size())
    set_current_dp_inner_rank(get_global_rank() % get_dp_world_size())

    assert kvargs["world_size"] % kvargs["args"].nnodes == 0, "world_size should be divided by nnodes"
    size_per_node = kvargs["world_size"] // kvargs["args"].nnodes
    device_id = kvargs["rank_id"] % size_per_node
    set_current_device_id(device_id)
    torch.cuda.set_device(device_id)
    if kvargs["world_size"] > 1:
        dist.init_process_group(
            "nccl",
            init_method=f'tcp://{kvargs["nccl_host"]}:{kvargs["nccl_port"]}',
            rank=kvargs["rank_id"],
            world_size=kvargs["world_size"],
        )
        # warmup nccl communicator
        _a = torch.zeros([1]).to(f"cuda:{device_id}")
        dist.all_reduce(_a)
        del _a


def set_global_rank(global_rank: int):
    set_environ("LIGHTLLM_GLOBAL_RANK", global_rank)


def get_global_rank():
    return int(get_environ("LIGHTLLM_GLOBAL_RANK"))


def set_global_world_size(world_size: int):
    set_environ("LIGHTLLM_GLOBAL_WORLD_SIZE", world_size)


def get_global_world_size():
    return int(get_environ("LIGHTLLM_GLOBAL_WORLD_SIZE"))


def set_dp_size(dp_size: int):
    """
    total dp num
    """
    set_environ("LIGHTLLM_DP_SIZE", dp_size)


def get_dp_size():
    return int(get_environ("LIGHTLLM_DP_SIZE"))


def set_dp_world_size(world_size: int):
    set_environ("LIGHTLLM_DP_WORLD_SIZE", world_size)


def get_dp_world_size():
    return int(get_environ("LIGHTLLM_DP_WORLD_SIZE"))


def set_current_dp_rank(rank: int):
    set_environ("LIGHTLLM_CURRENT_DP_RANK", rank)


def get_current_dp_rank():
    return int(get_environ("LIGHTLLM_CURRENT_DP_RANK"))


def set_current_dp_inner_rank(rank: int):
    set_environ("LIGHTLLM_CURRENT_DP_INNER_RANK", rank)


def get_current_dp_inner_rank():
    return get_environ("LIGHTLLM_CURRENT_DP_INNER_RANK")


def set_current_device_id(device_id: int):
    set_environ("LIGHTLLM_CURRENT_DEVICE_ID", device_id)


def get_current_device_id():
    return int(get_environ("LIGHTLLM_CURRENT_DEVICE_ID"))
