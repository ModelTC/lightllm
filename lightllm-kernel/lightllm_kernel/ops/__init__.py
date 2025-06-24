import importlib
import os
from pathlib import Path
from torch.utils.cpp_extension import load

PKG = "lightllm_kernel"
try:
    _C = importlib.import_module(f"{PKG}._C")
except ImportError:
    # raise ImportError("Cannot import compiled extension 'lightllm_kernel.ops'")
    repo_root = Path(__file__).resolve().parents[3]
    kernels_root = Path(__file__).resolve().parents[2]
    csrc_dir = kernels_root / "csrc"
    if not csrc_dir.exists():
        raise ImportError(
            "Cannot import compiled extension 'lightllm_kernel.ops' and no source "
            "directory (csrc/) found; please ensure you have run "
            "'cmake --install' or placed lightllm_kernel.ops.so on PYTHONPATH."
        )

    PROGRAM_NAME = "lightllm_kernel._C"
    EXTENSION_BUILD_DIR = "build"
    INCLUDE_DIR = "include"
    CUTLASS_DIR = "third-party/cutlass/include"

    sources = []
    file_names = []  # Store file names for printing
    for subdir, _, files in os.walk(csrc_dir):
        for file in files:
            if file.endswith((".cpp", ".cu")):
                sources.append(os.path.join(subdir, file))
                file_names.append(file)

    # Print all detected source file names
    print(f"{PROGRAM_NAME}: Detected source files:")
    for file_name in file_names:
        print(f"  - {file_name}")

    _C = load(
        name=PROGRAM_NAME,
        sources=sources,
        verbose=True,
        extra_include_paths=[
            os.path.join(kernels_root, INCLUDE_DIR),
            os.path.join(repo_root, CUTLASS_DIR),
        ],
        build_directory=os.path.join(kernels_root, EXTENSION_BUILD_DIR),
        with_cuda=True,
        extra_ldflags=["-lcuda", "-L/usr/local/cuda/lib64"],
        extra_cuda_cflags=[
            "-DNDEBUG",
            "-O3",
            "-use_fast_math",
            # A100
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_80,code=compute_80",
            # Ada / L40s / 4090
            "-gencode=arch=compute_89,code=sm_89",
            "-gencode=arch=compute_89,code=compute_89",
            # Hopper / H100 / H200
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
            "-gencode=arch=compute_90a,code=sm_90a",
        ],
        extra_cflags=["-O3"],
    )

meta_size = _C.meta_size
# 向外暴露 Python 端接口
from .fusion import pre_tp_norm_bf16, post_tp_norm_bf16, add_norm_quant_bf16_fp8, gelu_per_token_quant_bf16_fp8
from .norm import rmsnorm_bf16
from .allgather import (
    all_gather,
    allgather_dispose,
    init_custom_gather_ar,
    allgather_register_buffer,
    allgather_register_graph_buffers,
    allgather_get_graph_buffer_ipc_meta,
)
from .quant import per_token_quant_bf16_fp8, per_token_quant_bf16_int8
from .gemm import cutlass_scaled_mm_bias_ls
from .moe import grouped_topk
from .attention import group8_int8kv_flashdecoding_stage1, group_int8kv_decode_attention

__all__ = [
    "rmsnorm_bf16",
    "per_token_quant_bf16_fp8",
    "per_token_quant_bf16_int8",
    "pre_tp_norm_bf16",
    "post_tp_norm_bf16",
    "add_norm_quant_bf16_fp8",
    "gelu_per_token_quant_bf16_fp8",
    "cutlass_scaled_mm_bias_ls",
    "grouped_topk",
    "meta_size",
    "all_gather",
    "allgather_dispose",
    "init_custom_gather_ar",
    "allgather_register_buffer",
    "allgather_get_graph_buffer_ipc_meta",
    "allgather_register_graph_buffers",
    "group8_int8kv_flashdecoding_stage1",
    "group_int8kv_decode_attention",
]
