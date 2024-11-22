from lightllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


def get_world_size():
    return get_tensor_model_parallel_world_size()


def get_rank():
    return get_tensor_model_parallel_rank()
