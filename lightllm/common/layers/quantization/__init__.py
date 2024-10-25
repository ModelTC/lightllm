from .ppl_quant import PPLW4A16QuantizationMethod


QUANTIZATION_METHODS = {"ppl_w4a16": PPLW4A16QuantizationMethod}


def get_quantization_method(mode):
    if "triton_w8a16" in mode:
        return QUANTIZATION_METHODS["triton_w8a16"]()
    elif "triton_w4a16" in mode:
        return QUANTIZATION_METHODS["triton_w4a16"]()
    elif "lmdeploy_w4a16" in mode:
        return QUANTIZATION_METHODS["lmdeploy_w4a16"]()
    elif "ppl_w4a16" in mode:
        return QUANTIZATION_METHODS["ppl_w4a16"]()
    elif "flash_llm_w6a16" in mode:
        return QUANTIZATION_METHODS["flash_llm_w6a16"]()
    else:
        return None
