#include "ops_common.h"
#include "reduce/sm70.cuh"


namespace lightllm {
namespace ops {

using namespace lightllm;

template<int32_t TPB, int32_t N>
__global__ void device_gelu_per_token_quant_bf16_to_fp8(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M                  // Number of rows in the input tensor
) {
    constexpr int32_t VPT = 8;

    static_assert(N % 2 == 0, "N must be even.");
    static_assert(N % VPT == 0, "N must be a multiple of VPT.");

    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format
    const bf16x2_t one =  _float22bf162_rn(make_float2(1.0f, 1.0f));
    const bf16x2_t one_2 =  _float22bf162_rn(make_float2(0.5f, 0.5f));
    
    const bf16_t* _input = input + bid * N; // Input pointer for the group
    fp8_e4m3_t* _output  = output + bid * N; // Output pointer for the group

    fp32_t* _scales;
    _scales = scales + bid;

    // Local arrays for intermediate storage
    fp8x4_e4m3_t local_f8[VPT / 4];
    bf16x2_t local_bf16[VPT / 2];

    __shared__ bf16x2_t workspace[N / 2];

    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_bf16);
        //gelu
        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
            fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]); 
            tmp.x = erf(tmp.x * 0.7071067811f);
            tmp.y = erf(tmp.y * 0.7071067811f);
            bf16x2_t tan =  _float22bf162_rn(tmp);
            tan = __hadd2(tan, one);
            tan = __hmul2(tan, local_bf16[j]);
            tan = __hmul2(tan, one_2);
            local_bf16[j] = tan;
        }

        vec_copy<sizeof(bf16_t) * VPT>(local_bf16, workspace + (i >> 1));
        
        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
           fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]); 
           fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
           local_max = fmaxf(local_max, max);
        }
    }

    // Reduce the maximum value across the thread group
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace + (i >> 1), local_bf16);

        #pragma unroll
        for (int32_t j = 0; j < VPT/4; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_bf16[2 * j + 0]);
            fp32x2_t y = bf16x2_to_fp32x2(local_bf16[2 * j + 1]);
            fp32x4_t ret = make_float4(
                x.x * inv_scale,
                x.y * inv_scale,
                y.x * inv_scale,
                y.y * inv_scale
            );
            local_f8[j] = fp8x4_e4m3_t(ret);
        }

        vec_copy<sizeof(fp8_e4m3_t) * VPT>(local_f8, _output + i);
    }

    if(tid == 0){
        *_scales = scale;
    }
}


template<int32_t TPB>
__global__ void gelu_per_token_quant_bf16_to_fp8_vpt(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M,                  // Number of rows in the input tensor
    const int32_t N
) {
    constexpr int32_t VPT = 8;

    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format
    constexpr fp32_t sqrt_2_over_pi = 0.7978845608028654f;
    constexpr fp32_t coeff = 0.044715f;
    
    const bf16_t* _input = input + bid * N; // Input pointer for the group
    fp8_e4m3_t* _output  = output + bid * N; // Output pointer for the group

    fp32_t* _scales;
    _scales = scales + bid;

    // Local arrays for intermediate storage
    fp8x4_e4m3_t local_f8[VPT / 4];
    bf16x2_t local_bf16[VPT / 2];

    extern __shared__ bf16x2_t workspace[];

    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_bf16);

        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
            fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]); 

           fp32_t tanh_arg1 = sqrt_2_over_pi * (tmp.x + coeff * tmp.x * tmp.x * tmp.x);
           fp32_t tanh_arg2 = sqrt_2_over_pi * (tmp.y + coeff * tmp.y * tmp.y * tmp.y);
           tmp.x = 0.5f * tmp.x * (1.0f + tanhf(tanh_arg1));
           tmp.y = 0.5f * tmp.y * (1.0f + tanhf(tanh_arg2));

           local_bf16[j] = _float22bf162_rn(tmp);
        }

        vec_copy<sizeof(bf16_t) * VPT>(local_bf16, workspace + (i >> 1));

        // Compute the max for the VPT elements.
        #pragma unroll
        for(int32_t j = 0; j< VPT/2; j++){
            fp32x2_t tmp = bf16x2_to_fp32x2(local_bf16[j]);
            fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
            local_max = fmaxf(local_max, max);
        }
    }

    // Reduce the maximum value across the thread group
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace + (i >> 1), local_bf16);

        #pragma unroll
        for (int32_t j = 0; j < VPT/4; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_bf16[2 * j + 0]);
            fp32x2_t y = bf16x2_to_fp32x2(local_bf16[2 * j + 1]);
            fp32x4_t ret = make_float4(
                x.x * inv_scale,
                x.y * inv_scale,
                y.x * inv_scale,
                y.y * inv_scale
            );
            local_f8[j] = fp8x4_e4m3_t(ret);
        }

        vec_copy<sizeof(fp8_e4m3_t) * VPT>(local_f8, _output + i);
    }

    if(tid == 0){
        *_scales = scale;
    }
}


template<int32_t TPB>
__global__ void gelu_per_token_quant_bf16_to_fp8_general(
    const bf16_t* __restrict__ input,  // Input tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M,                  // Number of rows in the input tensor
    const int32_t N
) {
    const int32_t bid = blockIdx.x;
    const int32_t tid = threadIdx.x;
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format
    constexpr fp32_t sqrt_2_over_pi = 0.7978845608028654f;
    constexpr fp32_t coeff = 0.044715f;
    
    const bf16_t* _input = input + bid * N; // Input pointer for the group
    fp8_e4m3_t* _output  = output + bid * N; // Output pointer for the group

    fp32_t* _scales;
    _scales = scales + bid;

    extern __shared__ bf16_t workspace_[];

    fp32_t local_max = -FLT_MAX;
  
    for (int32_t i = tid; i < N; i += TPB) {
        fp32_t tmp = cvt_bf16_f32(_input[i]);
        fp32_t tanh_arg = sqrt_2_over_pi * (tmp + coeff * tmp * tmp * tmp);
        tmp = 0.5f * tmp * (1.0f + tanhf(tanh_arg));
        local_max = fmaxf(local_max, fabsf(tmp));
        workspace_[i] = cvt_f32_bf16(tmp);
    }

    // Reduce the maximum value across the thread group
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid; i < N; i += TPB) {
        // Load the previously stored vectorized data from shared memory.
        fp32_t x = cvt_bf16_f32(workspace_[i]);
        // Apply normalization: multiply by inv_norm and then scale by the weight.
        fp32_t ret = x * inv_scale;
        _output[i] = fp8_e4m3_t(ret);
    }

    if(tid == 0){
        *_scales = scale;
    }
}

void gelu_per_token_quant_bf16_fp8 (
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
            device_gelu_per_token_quant_bf16_to_fp8<64, 16>
            <<<blocks, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 32:
            device_gelu_per_token_quant_bf16_to_fp8<64, 32>
            <<<blocks, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 64:
            device_gelu_per_token_quant_bf16_to_fp8<64, 64>
            <<<blocks, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 512:
            device_gelu_per_token_quant_bf16_to_fp8<64, 512>
            <<<blocks, 64, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;

        case 1024:
            device_gelu_per_token_quant_bf16_to_fp8<128, 1024>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 2048:
            device_gelu_per_token_quant_bf16_to_fp8<128, 2048>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 3200:
            device_gelu_per_token_quant_bf16_to_fp8<128, 3200>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 4096:
            device_gelu_per_token_quant_bf16_to_fp8<256, 4096>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        case 12800:
            device_gelu_per_token_quant_bf16_to_fp8<256, 12800>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_input),
                PTR<fp8_e4m3_t>(output),
                PTR<fp32_t>(contiguous_scales),
                M
            );
            break;
        default: {
            static constexpr int32_t TPB = 128;
            int32_t sharedmem = N / 2 * sizeof(bf16x2_t);
            if (N % 8 == 0) {
                gelu_per_token_quant_bf16_to_fp8_vpt<128>
                <<<blocks, TPB, sharedmem, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_input),
                    PTR<fp8_e4m3_t>(output),
                    PTR<fp32_t>(contiguous_scales),
                    M, N
                );
            }
            else {
                gelu_per_token_quant_bf16_to_fp8_general<128>
                <<<blocks, TPB, sharedmem, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_input),
                    PTR<fp8_e4m3_t>(output),
                    PTR<fp32_t>(contiguous_scales),
                    M, N
                );
            }
        }
    }
    return ;
}

} // namespace ops
} // namespace lightllm