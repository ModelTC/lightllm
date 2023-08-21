import torch
torch.multiprocessing.set_start_method('spawn', force=True) # Fork start method will cause CUDA re-initialization error
import os
import json

def get_total_free_gpu_memory(tp):
    """
    Returns the total amount of free memory available on all GPUs, in Gigabytes.
    """
    devices = min(tp, torch.cuda.device_count())
    total_free = 0
    for i in range(devices):
        total_free += torch.cuda.mem_get_info(i)[0]
    total_free = total_free / (1024 ** 3)
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

def get_kv_cache_size(model_dir):
    """
    Returns the size of the kv cache for a single token, in Gigabytes.
    """
    # Read from config.json
    config_path = os.path.join(model_dir, 'config.json')
    assert os.path.exists(config_path), "config.json not found in model directory."
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        hidden_size = config['hidden_size']
        layer_num = config['num_hidden_layers']
        num_attention_heads = config['num_attention_heads']
        num_key_value_heads = config.get('num_key_value_heads', num_attention_heads) # Models may not be using GQA
        dtype = config.get('torch_dtype', 'float16') # TODO: dtype may not be specified in config.json, should we load weights to check?
    except:
        raise Exception("Error reading config.json when trying to determine max_total_token_num. Please manually specify max_total_token_num in startup arguments.")
    dtype_size = torch.empty(0, dtype=getattr(torch, dtype)).element_size()
    kv_cache_size = hidden_size * dtype_size * 2 * layer_num / num_attention_heads * num_key_value_heads / (1024 ** 3)
    return kv_cache_size

def calc_max_total_token_num(tp, weight_dir, mem_fill_rate=0.8):
    """
    Calculate the max total token num that can be supported by the model.
    """
    kv_cache_size = get_kv_cache_size(weight_dir)
    max_token_num = (get_total_free_gpu_memory(tp)-get_total_weight_size(weight_dir)) * mem_fill_rate / kv_cache_size
    return int(max_token_num)