# LightLLM-Kernel

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LightLLM-Kernel is a high-performance CUDA kernel library powering the LightLLM inference system. It provides optimized GPU implementations for critical operations in large language model (LLM) inference, delivering significant performance improvements through carefully crafted CUDA kernels.

## Project Overview

LightLLM-Kernel serves as the computational backbone for LightLLM framework, offering:
- **Custom CUDA Kernels**: Highly optimized implementations for transformer-based model operations
- **Memory Efficiency**: Reduced memory footprint through advanced quantization techniques
- **Scalability**: Support for large model architectures including MoE (Mixture-of-Experts) models

## Key Features

### Core Modules
| Module       | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| **Attention** | Optimized Multi-Head Attention kernels with fused QKV operations and efficient softmax         |
| **MoE**       | Expert routing and computation kernels for Mixture-of-Experts architectures                    |
| **Quant**     | Low-precision quantization support (INT8/INT4) for weights and activations                      |
| **Extensions**| Continuous expansion of optimized operations for emerging model architectures                   |

## Installation

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