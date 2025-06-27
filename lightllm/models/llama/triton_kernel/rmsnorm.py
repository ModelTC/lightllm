import os
import torch
import triton
import triton.language as tl
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager


@triton.jit
def _rms_norm_low_accuracy_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    x_stride0,  # how much to increase the pointer when moving by 1 row
    x_stride1,
    y_stride0,
    y_stride1,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * y_stride0
    X += row * x_stride0
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols * x_stride1, mask=cols < N, other=0.0).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols * y_stride1, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm_forward_low_accuracy(x: torch.Tensor, weight, eps, use_custom_tensor_mananger: bool = False):
    # allocate output
    if use_custom_tensor_mananger:
        shape = x.shape
        dtype = x.dtype
        device = x.device
        y = g_cache_manager.alloc_tensor(shape, dtype, device=device)
    else:
        y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.view(-1, x.shape[-1])
    y_arg = y.view(-1, x.shape[-1])
    assert y.data_ptr() == y_arg.data_ptr()
    M, N = x_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # print("BLOCK_SIZE:", BLOCK_SIZE)
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    num_warps = triton.next_power_of_2(num_warps)
    if BLOCK_SIZE > 16384:
        BLOCK_SIZE = 16384
    # enqueue kernel
    _rms_norm_low_accuracy_kernel[(M,)](
        x_arg,
        y_arg,
        weight,
        x_arg.stride(0),
        x_arg.stride(1),
        y_arg.stride(0),
        y_arg.stride(1),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


@triton.jit
def _rms_norm_high_accuracy_kernel(
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


def rmsnorm_forward_high_accuracy(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5, use_custom_tensor_mananger: bool = False
):
    """Rms norm."""

    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"

    origin_shape = hidden_states.shape
    hidden_dim = weight.shape[0]
    assert hidden_dim == origin_shape[-1], f"hidden_dim {hidden_dim} != {origin_shape[-1]}"

    rows = hidden_states.numel() // hidden_dim
    if hidden_states.dim() == 3:  # (bs, seq_len, hidden_dim)
        hidden_states = hidden_states.view(rows, hidden_dim)

    in_row_stride, in_col_stride = hidden_states.stride(0), hidden_states.stride(1)

    BLOCK_N = triton.next_power_of_2(hidden_dim)
    if use_custom_tensor_mananger:
        shape = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        output = g_cache_manager.alloc_tensor(shape, dtype, device=device)
    else:
        output = torch.empty_like(hidden_states)

    out_row_stride, out_col_stride = output.stride(0), output.stride(1)
    grid = (rows,)
    _rms_norm_high_accuracy_kernel[grid](
        hidden_states,
        weight,
        output,
        in_row_stride,
        in_col_stride,
        out_row_stride,
        out_col_stride,
        eps=eps,
        N_COLS=hidden_dim,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(origin_shape)


def torch_rms_norm(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def test_rms_norm(M, N, dtype, eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device="cuda")
    # forward pass
    y_tri = rmsnorm_forward_low_accuracy(x, weight, eps)
    y_tri_high_acc = rmsnorm_forward_high_accuracy(x, weight, eps)
    y_ref = torch_rms_norm(x.to(torch.float32), weight.to(torch.float32), eps).to(dtype)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype, y_tri_high_acc.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    print("max delta:", torch.max(torch.abs(y_tri_high_acc - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    return


use_high_acc = os.getenv("RMSNORM_HIGH_ACCURACY", "False").upper() in ["ON", "TRUE", "1"]

if use_high_acc:
    rmsnorm_forward = rmsnorm_forward_high_accuracy
else:
    rmsnorm_forward = rmsnorm_forward_low_accuracy
