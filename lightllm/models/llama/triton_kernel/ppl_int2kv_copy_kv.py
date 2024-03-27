import torch
import triton.language as tl


@torch.no_grad()
def destindex_copy_int2kv(K, DestLoc, Out, Out_scale):
    head_dim = K.shape[2]
    quant_group_dim = 8
    assert head_dim % quant_group_dim == 0, "error head dim, can not been supported to copy quant kv"
    group_size = head_dim // quant_group_dim
    group_dim = quant_group_dim

    K = K.view((K.shape[0], K.shape[1], group_size, group_dim))
    Out = Out.view(
        Out.shape[0], Out.shape[1], group_size, group_dim // 4
    )  # OUt 是 int8 类型， 四个int2组一个int8，所以 group_dim // 4

    from lightllm_ppl_int2kv_flashdecoding_kernel import group8_copy_int2_kv

    group8_copy_int2_kv(Out, Out_scale, K, DestLoc, 4)
