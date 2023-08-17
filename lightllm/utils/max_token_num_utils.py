import torch
torch.multiprocessing.set_start_method('spawn', force=True) # Fork start method will cause CUDA re-initialization error
import os

def get_total_free_gpu_memory(tp):
    """
    Returns the total amount of free memory available on all GPUs, in Gigabytes.
    """
    devices = min(tp, torch.cuda.device_count())
    total_free = 0
    for i in range(devices):
        total_free += torch.cuda.mem_get_info(i)[0]
    total_free = total_free / (1024 ** 3)
    torch.cuda.close()
    return total_free

def get_total_weight_size(weight_dir):
    """
    Returns the total size of all parameters in the model, in Gigabytes.
    """
    total_size = 0
    files = os.listdir(weight_dir)
    candidate_files = list(filter(lambda x : x.endswith('.safetensors'), files))
    if len(candidate_files) == 0:
        candidate_files = list(filter(lambda x : x.endswith('.bin'), files))
    assert len(candidate_files) != 0, "can only support pytorch tensor and safetensors format for weights."
    for file in candidate_files:
        total_size += os.path.getsize(os.path.join(weight_dir, file))
    total_size = total_size / (1024 ** 3)
    return total_size

def calc_max_total_token_num(tp, weight_dir, mem_fill_rate=0.8, kv_cache_size=0.000488281):
    """
    Calculate the max total token num that can be supported by the model.
    """
    max_token_num = (get_total_free_gpu_memory(tp)-get_total_weight_size(weight_dir)) * mem_fill_rate / kv_cache_size
    return int(max_token_num)