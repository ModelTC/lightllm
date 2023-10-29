import argparse
import torch
from torch import nn
from loguru import logger
from transformers import AutoModelForCausalLM, AutoConfig
from lightllm.common.basemodel.triton_kernel.quantize_gemm_int8 import quantize_int8
from lightllm.common.basemodel.triton_kernel.dequantize_gemm_int4 import quantize_int4
from functools import partial


class RealQuantLinear(nn.Module):
    quantize_func_dict = {
        "int8weight": quantize_int8,
        "int4weight": partial(quantize_int4, group_size=128),
    }

    def __init__(self, weight, bias, scales, zeros):
        super().__init__()
        self.register_buffer("weight", weight)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.register_buffer("scales", scales)

        if zeros is not None:
            self.register_buffer("zeros", zeros)
        else:
            self.zero = None

    @classmethod
    @torch.no_grad()
    def w_q(cls, weight, mode):
        weight = weight.cuda()
        weight = weight.transpose(0, 1)
        weight, scales, zeros = cls.quantize_func_dict[mode](weight.data.clone())
        weight = weight.transpose(1, 0).contiguous().cpu()
        if scales.dim() == 1:
            scales = scales.contiguous().cpu()
        elif scales.dim() == 2:
            scales = scales.transpose(1, 0).contiguous().cpu()
        else:
            raise NotImplementedError
        if zeros is not None:
            if zeros.dim() == 1:
                zeros = zeros.contiguous().cpu()
            elif scales.dim() == 2:
                zeros = zeros.transpose(1, 0).contiguous().cpu()
            else:
                raise NotImplementedError
        return weight, scales, zeros

    @classmethod
    @torch.no_grad()
    def new(cls, module, mode):
        assert isinstance(module, torch.nn.Linear)
        weight, scales, zeros = cls.w_q(module.weight.data.clone(), mode)

        if module.bias is not None:
            bias = module.bias.clone()
        else:
            bias = None

        new_module = cls(weight, bias, scales, zeros)
        new_module.in_features = module.in_features
        new_module.out_features = module.out_features
        new_module.weight_shape = weight.shape
        new_module.weight_dtype = weight.dtype
        new_module.scales_shape = scales.shape
        new_module.scales_dtype = scales.dtype

        if zeros is not None:
            new_module.zeros_shape = zeros.shape
            new_module.zeros_dtype = zeros.dtype
        else:
            new_module.zeros_shape = None
            new_module.zeros_dtype = None

        return new_module

    def __repr__(self):
        return (
            f"RealQuantLinear("
            + f"in_features={self.in_features}, "
            + f"out_features={self.out_features}, "
            + f"bias={self.bias is not None}, "
            + f"weight_shape={self.weight_shape}, "
            + f"weight_dtype={self.weight_dtype}, "
            + f"scales_shape={self.scales_shape}, "
            + f"scales_dtype={self.scales_dtype}, "
            + f"zeros_shape={self.zeros_shape}, "
            + f"zeros_dtype={self.zeros_dtype})"
        )


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument(
    "--mode", type=str, choices=["int8weight", "int4weight"], required=True
)
parser.add_argument("--model_output", type=str, required=True)
args = parser.parse_args()

model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    config=model_config,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)

blocks = model.model.layers
for i in range(len(blocks)):
    logger.info(f"Replace block index: {i+1}/{len(blocks)}")
    block = blocks[i]
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue

        module = RealQuantLinear.new(m, args.mode)

        parent_name = name.rsplit(".", 1)[0]
        parent = block.get_submodule(parent_name)

        logger.info(
            f"Replacing {name} with new module; parent: {parent_name}, child's name: {name[len(parent_name) + 1:]}"
        )
        setattr(parent, name[len(parent_name) + 1 :], module)


logger.info(f"The Replaced model: {model}")

model.save_pretrained(args.model_output)
