## todo
1. 量化算法根据activation统计的fp_indices, 因此对于q/k/v, gate/up算子是可以合并计算的
2. tensor-parallel中按行拆分的`o_proj`和`down_proj`无法满足算法要求(限制: 1. wReduced需重新计算(将rank上的int_weight反量化后累加) 2. 当前asym_quantize算子不支持 奇数dim的输入)
3. 算子融合
    * QUIK qlinear计算流程:  split and reorder outliers to int_x and fp_x -> quantize(int_x) -> matmul(fp_x) -> 
    int_matmul(int_x) -> dequantize(fp_x, int_x, scales, etc..).
    * 融合方式:
        * 将 split reorder quantize 融合到rms_norm中
        * 将 matmul(fp_x) 和 dequantize融合到int_matmul(int) 中