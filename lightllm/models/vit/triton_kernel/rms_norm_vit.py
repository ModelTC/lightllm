import torch
import triton
import triton.language as tl
from torch import Tensor
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    in_row_stride: tl.constexpr,
    in_col_stride: tl.constexpr,
    out_row_stride: tl.constexpr,
    out_col_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS, other=0.0)

    x_ptr = input + prog_id * in_row_stride
    x = tl.load(x_ptr + offsets * in_col_stride, mask=offsets < N_COLS, other=0.0)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * out_row_stride
    tl.store(out_ptr + offsets * out_col_stride, out, mask=offsets < N_COLS)


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5, use_custom_tensor_mananger: bool = False):
    """Rms norm."""
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    in_row_stride, in_col_stride = hidden_states.stride(-2), hidden_states.stride(-1)

    BLOCK_N = triton.next_power_of_2(feat_size)
    if use_custom_tensor_mananger:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        output = g_cache_manager.alloc_tensor(shape, dtype, device=device)
    else:
        output = torch.empty_like(hidden_states)

    out_row_stride, out_col_stride = output.stride(-2), output.stride(-1)
    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        output,
        in_row_stride,
        in_col_stride,
        out_row_stride,
        out_col_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return output


def test():
    def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float):
        var = (x.float() ** 2).mean(dim=-1, keepdim=True)
        y = x.float() / torch.sqrt(var + eps)
        return (y * weight).to(x.dtype)

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    seq_len, hidden_size = 1025, 3200
    eps = 1e-5

    weight = torch.randn(hidden_size, device=device, dtype=dtype)

    hs_contig = torch.randn(seq_len, hidden_size, device=device, dtype=dtype).contiguous()
    out_contig = rms_norm(hs_contig, weight, eps)
    ref_contig = _rms_norm_ref(hs_contig, weight, eps)
    assert torch.allclose(out_contig, ref_contig, atol=1e-3, rtol=1e-3), "contiguous check failed"

    buf = torch.randn(seq_len * 2, hidden_size, device=device, dtype=dtype)
    hs_noncontig = buf[::2]
    assert not hs_noncontig.is_contiguous()

    out_noncontig = rms_norm(hs_noncontig, weight, eps)
    ref_noncontig = _rms_norm_ref(hs_noncontig, weight, eps)
    assert torch.allclose(out_noncontig, ref_noncontig, atol=1e-3, rtol=1e-3), "non-contiguous check failed"

    print("both contiguous and non-contiguous tests pass")
