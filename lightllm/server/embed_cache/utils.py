import torch
import numpy as np
from io import BytesIO
import multiprocessing.shared_memory as shm


def tensor2bytes(t: torch.Tensor):
    if t.dtype == torch.float32:
        t = t.cpu().numpy().tobytes()
    else:
        t = t.cpu().to(torch.uint16).numpy().tobytes()
    return t


def bytes2tensor(b, torch_dtype=torch.bfloat16):
    if torch_dtype == torch.float32:
        arr_loaded = np.frombuffer(b, dtype=np.float32)
    else:
        arr_loaded = np.frombuffer(b, dtype=np.uint16)
    return torch.from_numpy(arr_loaded).to(torch_dtype)


def create_shm(name, data):
    try:
        data_size = len(data)
        shared_memory = shm.SharedMemory(name=name, create=True, size=data_size)
        mem_view = shared_memory.buf
        mem_view[:data_size] = data
    except FileExistsError:
        print("Warning create shm {} failed because of FileExistsError!".format(name))


def read_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    data = shared_memory.buf.tobytes()
    return data


def free_shm(name):
    shared_memory = shm.SharedMemory(name=name)
    shared_memory.close()
    shared_memory.unlink()


def get_shm_name_data(uid):
    return str(uid) + "-data"


def get_shm_name_embed(uid):
    return str(uid) + "-embed"
