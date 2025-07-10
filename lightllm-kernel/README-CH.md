# LightLLM-Kernel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

lightllm-kernel 是大模型推理系统 LightLLM 的 CUDA 算子库。它提供了在大型模型推理过程中所需的一系列自定义 GPU 运算算子，以加速关键步骤的计算。

## 功能列表

| Module       | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| **Attention** | Optimized Multi-Head Attention kernels with fused QKV operations and efficient softmax         |
| **MoE**       | Expert routing and computation kernels for Mixture-of-Experts architectures                    |
| **Quant**     | Low-precision quantization support (INT8/INT4) for weights and activations                      |
| **Extensions**| Continuous expansion of optimized operations for emerging model architectures                   |

## 安装方法

lightllm_kernel 提供了静态编译以及JIT（Just-In-Time）动态编译的安装方式。推荐使用静态编译安装以获得最佳性能，同时也支持开发者使用可编辑安装进行开发调试。

### System Requirements
- NVIDIA GPU with Compute Capability ≥ 7.0 (Volta+)
- CUDA 11.8 or higher
- Python 3.8+

### Installation Methods

#### Static Compilation (Recommended)
```bash
git clone https://github.com/YourUsername/lightllm_kernel.git
cd lightllm_kernel
make build
# Alternative using pip
pip install .
```

## 贡献指南
欢迎社区开发者为 lightllm_kernel 做出贡献！如果您计划新增自定义算子或改进现有功能，请参考以下指南：
- 新增算子实现：在 csrc/ 目录下添加您的 CUDA/C++ 源码文件，添加时建议参考现有算子的代码风格和结构。
- 注册Python接口：在 csrc/ops_bindings.cpp中，将新增的算子通过 PyBind11 或 TORCH_LIBRARY 等机制注册到 Python 接口。
- 导出算子到Python模块：在lightllm_kernel/ops/__init__.py只添加相应的导出代码，使新算子包含在 lightllm_kernel.ops 模块中。
- 本地测试：开发完成后，请在本地对您的更改进行测试。您可以编译安装新的版本并编写简单的脚本调用新算子，检查其功能和性能是否符合预期。如果项目附带了测试用例，也请运行所有测试确保不引入回归。
- 