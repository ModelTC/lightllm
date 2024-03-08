## todo
1. 量化算法根据activation统计的fp_indices, 因此对于q/k/v, gate/up算子是可以合并计算的
2. tensor-parallel中行拆分的 o_proj和down_proj无法满足算法要求(限制: 1. wReduced计算需独立 2. 当前asym_quantize算子不支持 奇数length的输入)