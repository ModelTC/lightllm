import importlib
import os
from pathlib import Path
from torch.utils.cpp_extension import load

PKG = "lightllm_kernel"
try:
    _C = importlib.import_module(f"{PKG}._C")
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    csrc_dir = repo_root / "csrc"
    if not csrc_dir.exists():
        raise ImportError(
            "Cannot import compiled extension 'lightllm_kernel.ops' and no source "
            "directory (csrc/) found; please ensure you have run "
            "'cmake --install' or placed lightllm_kernel.ops.so on PYTHONPATH."
        )

    sources = (
        [str(p) for p in (csrc_dir / "moe").glob("*.cpp")]
        + [str(p) for p in (csrc_dir / "moe").glob("*.cu")]
        + [str(csrc_dir / "ops_bindings.cpp")]
    )

    _C = load(
        name="lightllm_kernel._C",
        sources=sources,
        verbose=True,
        extra_cuda_cflags=[
            # A100
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_80,code=compute_80",
            # Ada / L40s / 4090
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_89,code=compute_89",
            # Hopper / H100 / H200
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
        ],
    )

# 向外暴露 Python 端接口
grouped_topk = _C.grouped_topk
__all__ = ["grouped_topk"]
