# bench_tp_norm_tp4.py
import os
import torch
import torch.distributed as dist
from types import SimpleNamespace

from lightllm_kernel.ops import (
    rmsnorm_bf16,
    pre_tp_norm_bf16,
    post_tp_norm_bf16,
)


def init_dist():
    dist.init_process_group("nccl", init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def tp_norm_cuda(x, w, cfg):
    if cfg.tp_world == 1:
        return rmsnorm_bf16(x, w, cfg.eps)

    var_local = pre_tp_norm_bf16(x)
    dist.all_reduce(var_local, op=dist.ReduceOp.SUM)
    return post_tp_norm_bf16(x, w, var_local, cfg.global_embed, cfg.eps)


def tp_norm_ref(x, w, cfg):
    x32 = x.to(torch.float32)
    var = x32.pow(2).sum(-1, keepdim=True)
    if cfg.tp_world > 1:
        dist.all_reduce(var, op=dist.ReduceOp.SUM)
    x32 = x32 * torch.rsqrt(var / cfg.global_embed + cfg.eps)
    return (w.to(torch.float32) * x32).to(x.dtype)


def bench(fn, tag, x, w, cfg, iters=200):
    for _ in range(20):
        fn(x, w, cfg)
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(True)
    t1 = torch.cuda.Event(True)
    t0.record()
    for _ in range(iters):
        fn(x, w, cfg)
    t1.record()
    torch.cuda.synchronize()
    ms = t0.elapsed_time(t1) / iters

    ref = tp_norm_ref(x, w, cfg).to(torch.float32)
    out = fn(x, w, cfg).to(torch.float32)
    mse = (out - ref).pow(2).mean().item()
    err = (out - ref).abs().max().item()

    if dist.get_rank() == 0:
        print(f"{tag:18s}| {ms:6.3f} ms | MSE {mse:.3e} | MaxErr {err:.3e}")


if __name__ == "__main__":
    rank, world = init_dist()

    tp_world = 4
    pad_heads, dim_h = 32, 1024
    local_embed = pad_heads * dim_h
    global_embed = local_embed * tp_world
    tokens = 2048
    eps = 1e-6

    x = torch.randn(tokens, local_embed, device=f"cuda:{rank}", dtype=torch.bfloat16)
    w = torch.randn(local_embed, device=f"cuda:{rank}", dtype=torch.bfloat16)

    cfg = SimpleNamespace(tp_world=tp_world, global_embed=global_embed, eps=eps)

    if rank == 0:
        print(
            f"tp={tp_world}, tokens={tokens}, local_embed={local_embed}, " f"global_embed={global_embed}, dtype=bf16\n"
        )
    dist.barrier()

    bench(tp_norm_ref, "torch_ref", x, w, cfg)
    bench(tp_norm_cuda, "cuda_kernel", x, w, cfg)

    dist.destroy_process_group()
# python -m torch.distributed.run --nproc_per_node=4 bench_tp_norm.py
