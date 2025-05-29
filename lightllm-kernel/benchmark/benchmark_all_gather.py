# benchmark_custom_allgather.py
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

# 导入扩展里的 API
from lightllm_kernel.ops import (
    init_custom_gather_ar,
    all_gather as custom_all_gather,
    allgather_dispose,
    meta_size,
)


def run(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)

    batch, dim = 32, 512
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        local = torch.randn(batch, dim, device=rank, dtype=dtype)
        fake_ptrs = [0] * world  # 简单占位
        handle = init_custom_gather_ar(fake_ptrs, local, rank, full_nvlink=False)

        out_custom = torch.empty(world * batch, dim, device=rank, dtype=dtype)

        # 预热
        for _ in range(10):
            custom_all_gather(handle, local, out_custom, 0, 0)
        torch.cuda.synchronize()

        # 计时：自定义
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(100):
            custom_all_gather(handle, local, out_custom, 0, 0)
        end.record()
        torch.cuda.synchronize()
        t_custom = start.elapsed_time(end) / 100  # ms

        # 计时：torch
        gathered = [torch.empty_like(local) for _ in range(world)]
        torch.cuda.synchronize()
        start.record()
        for _ in range(100):
            dist.all_gather(gathered, local)
        end.record()
        torch.cuda.synchronize()
        t_torch = start.elapsed_time(end) / 100  # ms

        # 精度对比
        custom_all_gather(handle, local, out_custom, 0, 0)
        ref = torch.cat(gathered, dim=0).to(torch.float32)
        diff = out_custom.to(torch.float32) - ref
        mse = diff.pow(2).mean().item()
        maxerr = diff.abs().max().item()

        if rank == 0:
            print(
                f"dtype={dtype:<10}  custom {t_custom:7.3f} ms   "
                f"torch {t_torch:7.3f} ms   "
                f"MSE {mse:.3e}   MaxErr {maxerr:.3e}"
            )

        allgather_dispose(handle)

    if rank == 0:
        print(f"meta_size() 返回 {meta_size()} 字节")
    dist.destroy_process_group()


if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    spawn(run, args=(gpus,), nprocs=gpus)
