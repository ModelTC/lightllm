import torch

import triton
import triton.language as tl

@triton.jit
def _silu_and_mul_kernel(
    input, output,
    stride_b,
    n_element,
    BLOCK_SIZE: tl.constexpr,
    ):
    tid = tl.program_id(0)
    input = input + tid * stride_b
    output = output + tid * stride_b // 2

    pid = tl.program_id(1)
    gate_block_start = pid * BLOCK_SIZE
    up_block_start = n_element + gate_block_start
    up_offsets = up_block_start + tl.arange(0, BLOCK_SIZE)
    gate_offsets = gate_block_start + tl.arange(0, BLOCK_SIZE)

    mask = gate_offsets < n_element

    up_ele = tl.load(input + up_offsets, mask=mask, other=0.0)
    gate_ele = tl.load(input + gate_offsets, mask=mask, other=0.0).to(tl.float32)

    gate_ele = gate_ele / (1 + tl.exp(-gate_ele))
    gate_ele = gate_ele.to(tl.float16)

    res = up_ele * gate_ele

    tl.store(output + gate_offsets, res.to(tl.float16), mask=mask)

def silu_and_mul_fwd(input):
    output_size = list(input.shape)
    output_size[-1] = output_size[-1] // 2
    output = torch.zeros(output_size, dtype=input.dtype, device=input.device)

    stride_b = input.stride(0)
    n_element = input.shape[-1] // 2

    grid = (input.shape[0], triton.cdiv(n_element, 128), )
    _silu_and_mul_kernel[grid](input, output, stride_b, input.shape[-1]//2, 128)
    return output

def torch_silu_and_mul(input: torch.Tensor):
    return torch.nn.functional.silu(input[:, 0:(input.shape[-1]//2)]) * input[:, (input.shape[-1]//2):]

def test_silu_and_mul(M, N, dtype, device='cuda'):
    # create data
    X = torch.load("../../../../up_gate_out.pt").to(device=device, dtype=dtype)

    # run
    y_tri = silu_and_mul_fwd(X)
    y_ref = torch_silu_and_mul(X)

    # compare
    print("type:", y_tri.dtype, y_ref.dtype)
    print("max delta:", torch.max(torch.abs(y_tri - y_ref)))
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    return

# test_silu_and_mul(16, 4096, torch.float16, device='cuda')