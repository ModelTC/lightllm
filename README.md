# Pre<span>$^3$</span>: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Paper](https://img.shields.io/badge/Pre3-OpenReview-red.svg)](https://openreview.net/pdf?id=g1aBeiyZEi)
<!--
[![arXiv](https://img.shields.io/badge/HarmoniCa-2410.01723-b31b1b)](https://arxiv.org/pdf/2410.01723)
[![GitHub Stars](https://img.shields.io/github/stars/ModelTC/HarmoniCa.svg?style=social&label=Star&maxAge=60)](https://github.com/ModelTC/HarmoniCa)
-->

[Junyi Chen](https://github.com/flyinglandlord), [Shihao Bai](https://github.com/shihaobai), [Zaijun Wang](https://github.com/hiworldwzj), [Siyu Wu](https://wusiyu.me/), Chuheng Du, [Hailong Yang](https://thomas-yang.github.io/), [Ruihao Gong📧](https://xhplus.github.io/), [Shengzhong Liu📧](https://liushengzhong1023.github.io/), [Fan Wu](https://cs.sjtu.edu.cn/~fwu/), [Guihai Chen](https://www.cs.sjtu.edu.cn/PeopleDetail.aspx?id=111)

(📧 denotes corresponding author.)

This is the official implementation of our paper introducing Pre$^3$, an efficient structured generation method for LLMs that optimizes LR(1) grammar processing. Existing approaches parse LR(1) grammars into pushdown automata (PDA), incurring runtime overhead for context-dependent token processing—particularly inefficient under large inference batches. In contrast, $\text{Pre}^3$ leverages precomputed prefix-conditioned edges during preprocessing to enable lightweight transitions and parallel processing. Additionally, we introduce a novel algorithm that transforms LR(1) transition graphs into deterministic pushdown automata (DPDA), eliminating runtime path exploration while maintaining minimal overhead. Seamlessly integrable with standard LLM inference frameworks, $\text{Pre}^3$ achieves up to 40% faster time per output token (TPOT) and 36% higher throughput in large batch size simulation experiments.

## News
<!--
* **May 28, 2025**: 🔥 We release our Python code presented in our paper. Have a try!
-->
* **May 15, 2025**: 🌟 Our paper has been accepted by ACL 2025 Main Conference! 🎉 Cheers!


## Overview

<p>
<img src= ./img/overview.png width="700"/>
</p>

Structured generation is crucial for LLM applications requiring formatted outputs like JSON or function calls, where constrained decoding ensures syntactic validity. Existing approaches based on LR(1) grammars or pushdown automata (PDA) face inherent inefficiencies: LR(1) methods incur computational overhead from context-dependent token processing, while PDA-based solutions suffer from non-deterministic transitions requiring runtime stack management. To address these limitations, we propose Pre³, a deterministic pushdown automaton (DPDA) framework that transforms LR(1) grammars through prefix-conditioned edges and cyclic-aware conversion. By precomputing all transitions and enabling parallel verification, Pre³ eliminates runtime exploration while maintaining grammatical constraints, providing an efficient solution for structured generation tasks. The framework integrates seamlessly with standard LLM inference pipelines.

## Quick Start

After cloning the repository, you can follow these steps to try our JSON structured generation.

### Requirements

With Python (=3.9) and PyTorch (>2.0) installed, execute the following command to install the  necessary packages and pre-trained models.

```bash
git checkout pre3-integrated
pip install -r requirements.txt
```

### Training

We'd like to provide the following script to launch the inference framework. More details about our method can be found in our paper and blog.

```bash
bash ./launch_lightllm.sh
```

### Inference

Here is the corresponding command for inference.

```bash
python test/format_out/test_pre3_constraint.py
```

## TODO

* A more robust and efficient implementation.

* Adapt to a wider variety of grammars.

## Acknowledgments

Our code was developed based on LightLLM, an efficient Python-based LLM inference framework. We thank the following projects for their pioneering work in structured generation that inspired our research:

- [SynCode](https://github.com/structuredllm/syncode) for its innovative approaches to LR(1)-grammar-constrained decoding.

- [Outlines](https://github.com/dottxt-ai/outlines) for its finite state machine-based structured generation techniques.

- [XGrammar](https://github.com/mlc-ai/xgrammar) for its breakthrough in context-free grammar processing and pushdown automata optimization.

<!--
## Citation

If you find our HarmoniCa useful or relevant to your research, please kindly cite our paper:

```
@inproceedings{
    anonymous2025harmonica,
    title={HarmoniCa: Harmonizing Training and Inference for Better Feature Caching in Diffusion Transformer Acceleration},
    author={Yushi Huang and Zining Wang and Ruihao Gong and Jing Liu and Xinjie Zhang and Jinyang Guo and Xianglong Liu and Jun Zhang},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
}
```
-->