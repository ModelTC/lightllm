from .ppl_quant import PPLW4A16QuantizationMethod, PPLW6A16QuantizationMethod
from .torchao_quant import (
    AOW4A16QuantizationMethod,
    AOW8A16QuantizationMethod,
    AOW8A8QuantizationMethod,
    AOFP8W8A16QuantizationMethod,
    AOFP6W6A16QuantizationMethod,
)
from .vllm_quant import vLLMw8a8QuantizationMethod

QUANTIZATION_METHODS = {
    "ppl_w4a16": PPLW4A16QuantizationMethod,
    "ppl_w6a16": PPLW6A16QuantizationMethod,
    "ao-int4wo": AOW4A16QuantizationMethod,
    "ao-int8wo": AOW8A16QuantizationMethod,
    "ao-w8a8": AOW8A8QuantizationMethod,
    "ao-fp8w8a16": AOFP8W8A16QuantizationMethod,
    "ao-fp6w6a16": AOFP6W6A16QuantizationMethod,
    "vllm-w8a8": vLLMw8a8QuantizationMethod,
}


def get_quantization_method(mode):
    if "triton_w8a16" in mode:
        return QUANTIZATION_METHODS["triton_w8a16"]()
    elif "triton_w4a16" in mode:
        return QUANTIZATION_METHODS["triton_w4a16"]()
    elif "lmdeploy_w4a16" in mode:
        return QUANTIZATION_METHODS["lmdeploy_w4a16"]()
    elif "ppl_w4a16" in mode:
        return QUANTIZATION_METHODS["ppl_w4a16"]()
    elif "ppl_w6a16" in mode:
        return QUANTIZATION_METHODS["ppl_w6a16"]()
    elif "flash_llm_w6a16" in mode:
        return QUANTIZATION_METHODS["flash_llm_w6a16"]()
    elif any(["ao" in m for m in mode]):
        ao_cfg = [m for m in mode if m.startswith("ao")][0]
        if "int4wo" in ao_cfg:
            group_size = int(ao_cfg.split("-")[-1])
            return QUANTIZATION_METHODS["ao-int4wo"](group_size=group_size)
        return QUANTIZATION_METHODS[ao_cfg]()
    elif any(["vllm" in m for m in mode]):
        vllm_cfg = [m for m in mode if m.startswith("vllm")][0]
        return QUANTIZATION_METHODS[vllm_cfg]()
    else:
        return None
