import os
from functools import lru_cache


@lru_cache(maxsize=None)
def set_current_device_id(device_id: int):
    os.environ["CURRENT_DEVICE_ID"] = str(device_id)


@lru_cache(maxsize=None)
def get_current_device_id():
    import torch

    if torch.cuda.is_available():
        default_device_id = torch.cuda.current_device()
        device_id = os.getenv("CURRENT_DEVICE_ID", default_device_id)
        return int(device_id)
    else:
        raise RuntimeError("Torch CUDA is not avaliable.")
