#include "ops_common.h"
#include "reduce/sm70.cuh"


namespace lightllm {
namespace ops {

using namespace lightllm;

// CUDA kernel for per token quantization from BF16 to INT8
template<int32_t TPB>
__global__ void device_per_token_quant_bf16_to_int8_general(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    int8_t* __restrict__ output,   // Output tensor in INT8 format
    fp32_t* __restrict__ scales,       // Output scales for each token
    const int64_t N
) {
    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t kINT8Max = 127.0f; // Maximum value representable in INT8 format
    
    const bf16_t* _input = input + bid * N; // Input pointer for the token
    int8_t* _output  = output + bid * N; // Output pointer for the token

    fp32_t* _scales;
    _scales = scales + bid;

    // Local arrays for intermediate storage
    int8_t local_int8;
    bf16_t local_bf16;

    extern __shared__ bf16_t workspace1[];

    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid; i < N; i += TPB) {
        local_bf16 = _input[i];
        workspace1[i] = local_bf16;

        fp32_t tmp = cvt_bf16_f32(local_bf16);
        local_max = fmaxf(local_max, fabsf(tmp));
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / kINT8Max;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid; i < N; i += TPB) {
        local_bf16 = workspace1[i];
        
        fp32_t tmp = cvt_bf16_f32(local_bf16);
        fp32_t x = tmp * inv_scale;
        local_int8 = float_to_int8_rn(x);

        _output[i] = local_int8;
    }

    if(tid == 0){
        *_scales = scale;
    }

}

// CUDA kernel for per token quantization from BF16 to INT8
template<int32_t TPB>
__global__ void device_per_token_quant_bf16_to_int8_vpt(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    int8_t* __restrict__ output,   // Output tensor in INT8 format
    fp32_t* __restrict__ scales,       // Output scales for each token
    const int32_t N
) {
    constexpr int32_t VPT = 8;

    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t kINT8Max = 127.0f; // Maximum value representable in INT8 format
    
    const bf16_t* _input = input + bid * N; // Input pointer for the token
    int8_t* _output  = output + bid * N; // Output pointer for the token

    fp32_t* _scales;
     _scales = scales + bid;

    // Local arrays for intermediate storage
    int8_t local_int8[VPT];
    bf16x2_t local_bf16[VPT / 2];

    extern __shared__ bf16x2_t workspace2[];

    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT FP16 elements from global memory (_X) into local vector (local_x).
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_bf16);

        vec_copy<sizeof(bf16_t) * VPT>(local_bf16, workspace2 + (i >> 1));

        // Compute the max for the VPT elements.
        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
            fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]);
            fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
            local_max = fmaxf(local_max, max);
        }
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / kINT8Max;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace2 + (i >> 1), local_bf16);

        #pragma unroll
        for (int32_t j = 0; j < VPT/2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_bf16[j]);

            int8_t a = float_to_int8_rn(x.x * inv_scale);
            int8_t b = float_to_int8_rn(x.y * inv_scale);
            
            local_int8[2 * j] = a;
            local_int8[2 * j + 1] = b;
        }

        vec_copy<sizeof(int8_t) * VPT>(local_int8, _output + i);
    }

    if(tid == 0){
        *_scales = scale;
    }
}



// CUDA kernel for per token quantization from BF16 to INT8
template<int32_t TPB, int32_t N>
__global__ void device_per_token_quant_bf16_to_int8(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    int8_t* __restrict__ output,   // Output tensor in INT8 format
    fp32_t* __restrict__ scales       // Output scales for each token
) {
    constexpr int32_t VPT = 8;

    static_assert(N % 2 == 0, "N must be even.");
    static_assert(N % VPT == 0, "N must be a multiple of VPT.");

    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t kINT8Max = 127.0f; // Maximum value representable in INT8 format
    
    const bf16_t* _input = input + bid * N; // Input pointer for the token
    int8_t* _output  = output + bid * N; // Output pointer for the token

    fp32_t* _scales;
    _scales = scales + bid;

    // Local arrays for intermediate storage
    int8_t local_int8[VPT];
    bf16x2_t local_bf16[VPT / 2];

    __shared__ bf16x2_t workspace[N / 2];

    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT FP16 elements from global memory (_X) into local vector (local_x).
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_bf16);

        vec_copy<sizeof(bf16_t) * VPT>(local_bf16, workspace + (i >> 1));

        // Compute the max for the VPT elements.
        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
            fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]);
            fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
            local_max = fmaxf(local_max, max);
        }
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / kINT8Max;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace + (i >> 1), local_bf16);

        #pragma unroll
        for (int32_t j = 0; j < VPT/2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_bf16[j]);

            int8_t a = float_to_int8_rn(x.x * inv_scale);
            int8_t b = float_to_int8_rn(x.y * inv_scale);

            local_int8[2 * j] = a;
            local_int8[2 * j + 1] = b;
        }

        vec_copy<sizeof(int8_t) * VPT>(local_int8, _output + i);
    }

    if(tid == 0){
        *_scales = scale;
    }
}


void per_token_quant_bf16_int8 (
    Tensor& output,
    const Tensor& input,
    Tensor& scales
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2-dimensional");
    TORCH_CHECK(input.scalar_type() == c10::kBFloat16, "Input must be BF16 type");

    Tensor contiguous_input = input.is_contiguous() ? input : input.contiguous();
    Tensor contiguous_scales = scales.is_contiguous() ? scales : scales.contiguous();

    const int64_t M = input.size(0);
    const int64_t N = input.size(1);

    const int32_t blocks = M;

    switch (N) {
        case 16:
            device_per_token_quant_bf16_to_int8<128, 16>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 32:
            device_per_token_quant_bf16_to_int8<128, 32>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 64:
            device_per_token_quant_bf16_to_int8<128, 64>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 512:
            device_per_token_quant_bf16_to_int8<128, 512>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 1024:
            device_per_token_quant_bf16_to_int8<128, 1024>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 3200:
            device_per_token_quant_bf16_to_int8<128, 3200>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 4096:
            device_per_token_quant_bf16_to_int8<128, 4096>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        case 12800:
            device_per_token_quant_bf16_to_int8<256, 12800>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<int8_t>(output),
                PTR<fp32_t>(contiguous_scales)
            );
            break;
        default: {
            static constexpr int TPB = 128;
            const int64_t shared_mem_size = N * sizeof(bf16_t);
            if (N % 8 == 0) {
                device_per_token_quant_bf16_to_int8_vpt<TPB>
                <<<blocks, TPB, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_input),
                    PTR<int8_t>(output),
                    PTR<fp32_t>(contiguous_scales),
                    N
                );
            } else {
                device_per_token_quant_bf16_to_int8_general<TPB>
                <<<blocks, TPB, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_input),
                    PTR<int8_t>(output),
                    PTR<fp32_t>(contiguous_scales),
                    N
                );
            }
        }
    }

    return;
}

} // namespace ops
} // namespace lightllm
