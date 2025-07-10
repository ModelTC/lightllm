import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

repo_root = Path(__file__).resolve().parents[1]
kernels_root = Path(__file__).resolve().parents[0]
csrc_dir = kernels_root / "csrc"
if not csrc_dir.exists():
    raise ImportError(
        "Cannot import compiled extension 'lightllm_kernel.ops' and no source "
        "directory (csrc/) found; please ensure you have run "
        "'cmake --install' or placed lightllm_kernel.ops.so on PYTHONPATH."
    )

PROGRAM_NAME = "lightllm_kernel._C"
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

ext_modules = [
    CUDAExtension(
        name=PROGRAM_NAME,
        sources=sources,
        libraries=["cuda"],
        library_dirs=["/lib/x86_64-linux-gnu"],
        extra_link_args=["-lcuda"],  # <-- 备选/补充方法
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-DNDEBUG",
                "-O3",
                "--use_fast_math",
                # A100 (compute_80)
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_80,code=compute_80",
                # A10 / other Ampere (compute_86)
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_86,code=compute_86",
                # L40s / 4090 (compute_89)
                "-gencode=arch=compute_89,code=sm_89",
                "-gencode=arch=compute_89,code=compute_89",
                # H100 (compute_90)
                "-gencode=arch=compute_90,code=sm_90",
                "-gencode=arch=compute_90,code=compute_90",
                "-gencode=arch=compute_90a, code=sm_90a",
            ],
        },
        include_dirs=[
            os.path.join(kernels_root, INCLUDE_DIR),
            os.path.join(repo_root, CUTLASS_DIR),
        ],
    )
]

setup(
    name="lightllm_kernel",
    packages=["lightllm_kernel", "lightllm_kernel.ops"],
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    package_dir={"ops": "ops"},
)
