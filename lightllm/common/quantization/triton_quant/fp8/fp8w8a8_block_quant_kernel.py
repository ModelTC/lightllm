import torch
import triton
import triton.language as tl
from lightllm.utils.dist_utils import get_current_device_id
from typing import Tuple


@triton.jit
def weight_quant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE) -> None:
    """
    Triton kernel for weight quantization to FP8 e4m3 format.
    
    Args:
        x_ptr: Input tensor pointer (float32)
        s_ptr: Output scale tensor pointer (float32)
        y_ptr: Output quantized tensor pointer (float8_e4m3fn)
        M: Number of rows
        N: Number of columns
        BLOCK_SIZE: Size of the processing block
    """
    
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n_blocks = tl.cdiv(N, BLOCK_SIZE)

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.abs(x))
    max_fp8e4m3_val = 448.0 
    scale = amax / (max_fp8e4m3_val + 1e-6) 
    y = (x / scale).to(y_ptr.dtype.element_ty)

    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n_blocks + pid_n, scale)


def weight_quant(
    x: torch.Tensor, 
    block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:

    if not x.is_contiguous():
        raise ValueError("Input tensor must be contiguous")
    
    if not x.is_cuda:
        x = x.cuda(get_current_device_id())
    
    if x.dim() not in (2, 3):
        raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D")
    
    # Handle 3D input by processing each batch
    if x.dim() == 3:
        batch_size, M, N = x.shape
        y_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        num_blocks_m = triton.cdiv(M, block_size)
        num_blocks_n = triton.cdiv(N, block_size)
        s_scales = torch.empty(
            (batch_size, num_blocks_m, num_blocks_n), 
            dtype=torch.float32, 
            device=x.device
        )
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), 
                           triton.cdiv(N, meta['BLOCK_SIZE']))
        
        for i in range(batch_size):
            weight_quant_kernel[grid](
                x[i], 
                s_scales[i], 
                y_quant[i], 
                M, 
                N, 
                BLOCK_SIZE=block_size
            )
        
        return y_quant, s_scales
    
    # Handle 2D input
    M, N = x.shape
    y_quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    num_blocks_m = triton.cdiv(M, block_size)
    num_blocks_n = triton.cdiv(N, block_size)
    s_scales = torch.empty(
        (num_blocks_m, num_blocks_n), 
        dtype=torch.float32, 
        device=x.device
    )
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), 
                        triton.cdiv(N, meta['BLOCK_SIZE']))
    
    weight_quant_kernel[grid](x, s_scales, y_quant, M, N, BLOCK_SIZE=block_size)
    
    return y_quant.t(), s_scales.t()
    
    