# mypy: allow-untyped-defs
import multiprocessing
import os
import threading
from multiprocessing.reduction import ForkingPickler
from multiprocessing.util import register_after_fork
from typing import Union

import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
from torch.multiprocessing.reductions import storage_from_cache, shared_cache, StorageWeakRef


def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # 因为接收进程在将 tensor 对应的 handle重新转化为指针的时候
    # 在其c++源码中会将当前显卡切换到storage_device再做操作，这样
    # 得到的指针可能不是接收进程当前上下文设备可以访问的，所以在这里
    # hack 修改了使用的 storage_device，这样后续tritonkernel同时
    # 访问几张显卡上的数据，进行p2p操作就不会出问题了。
    storage_device = torch.cuda.current_device()
    # If storage_handle is None, storage points to nullptr.
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        storage = storage_from_cache(storage_cls, (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(storage)
        else:
            # We already ref counting this Storage, but producer needs new ref-counters to be released.
            storage_cls._release_ipc_counter(ref_counter_handle, ref_counter_offset, device=storage_device)

    _storage = storage if isinstance(storage, torch.UntypedStorage) else storage._untyped_storage

    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # It is crucial for integer tensors to receive
        # the requires_grad=False as an argument in the constructor
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad

    return t
