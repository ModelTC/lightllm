from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = Path(__file__).parent

sources = [
    str(this_dir / "csrc" / "moe" / "grouped_topk_interface.cpp"),
    str(this_dir / "csrc" / "moe" / "grouped_topk.cu"),
    str(this_dir / "csrc" / "ops_bindings.cpp"),
]
print("---- sources for CUDAExtension ----")
for s in sources:
    print(s)
print("-----------------------------------")
ext_modules = [
    CUDAExtension(
        name="lightllm_kernel._C",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-gencode=arch=compute_90,code=sm_90",
                "-gencode=arch=compute_90,code=compute_90",
            ],
        },
        include_dirs=[str(this_dir / "include")],
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
