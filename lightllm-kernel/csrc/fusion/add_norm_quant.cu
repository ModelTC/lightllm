#include "ops_common.h"
#include "reduce/sm70.cuh"

namespace lightllm {
namespace ops {

using namespace lightllm;

template<int32_t TPB>
__global__ void device_add_norm_quant_bf16_general(
    bf16_t* __restrict__ input,  // Input tensor in BF16 format
    const bf16_t* __restrict__ residual, // Residual tensor in BF16 format
    const bf16_t* __restrict__ weight, // Weight tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M,                   // Number of rows in the input tensor
    const int32_t N,                   // Number of cols in the input tensor
    const fp32_t eps                   // Epsilon value for numerical stability
) {
    const fp32_t r_N = 1 / (fp32_t)N;       // Reciprocal of N.
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _input = input + bid * N;
    const bf16_t* _residual = residual + bid * N;
    fp8_e4m3_t* _output = output + bid * N;

    fp32_t* _scales;
     _scales = scales + bid;

    // Shared memory workspace to store data.
    extern __shared__ bf16_t workspace1[];

    // Local registers to hold data.
    bf16_t local_input;
    bf16_t local_residual;
    bf16_t local_w;
    bf16_t local_output;
    fp8_e4m3_t local_f8;
    

    // Each thread computes a partial sum of squares.
    fp32_t local_square_sum = 0.0f;
    for (int32_t i = tid; i < N; i += TPB) {
        local_input = _input[i];
        local_residual = _residual[i];

        fp32_t x = cvt_bf16_f32(local_input);
        fp32_t r = cvt_bf16_f32(local_residual);
        local_input = cvt_f32_bf16(x + r);
        fp32_t tmp = cvt_bf16_f32(local_input);
        local_square_sum += tmp * tmp;

        _input[i] = local_input;
        workspace1[i] = local_input;
    }

    const fp32_t reduced_square_sum = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum); 

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    const fp32_t mean_square = reduced_square_sum * r_N;
    const fp32_t inv_norm = rsqrtf(mean_square + eps);

    // Normalize each element using the computed normalization factor.
    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid; i < N; i += TPB) {
        local_input = workspace1[i];
        local_w = weight[i];

        fp32_t x = cvt_bf16_f32(local_input);
        fp32_t w = cvt_bf16_f32(local_w);

        fp32_t ret = x * inv_norm * w;
        local_output = cvt_f32_bf16(ret);
        fp32_t tmp = cvt_bf16_f32(local_output);
        local_max = fmaxf(local_max, fabsf(tmp));

        workspace1[i] = local_output;
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid; i < N; i += TPB) {
        local_output = workspace1[i];

        fp32_t tmp = cvt_bf16_f32(local_output);
        fp32_t ret = tmp * inv_scale;
        local_f8 = fp8_e4m3_t(ret);

        _output[i] = local_f8;
    }

    if(tid == 0){
        *_scales = scale;
    }
}



template<int32_t TPB>
__global__ void device_add_norm_quant_bf16_vpt(
    bf16_t* __restrict__ input,  // Input tensor in BF16 format
    const bf16_t* __restrict__ residual, // Residual tensor in BF16 format
    const bf16_t* __restrict__ weight, // Weight tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M,                   // Number of rows in the input tensor
    const int32_t N,                   // Number of cols in the input tensor
    const fp32_t eps                   // Epsilon value for numerical stability
) {
    constexpr int32_t VPT = 8;                // Number of FP16 values processed per thread.
    const fp32_t r_N = 1 / (fp32_t)N;       // Reciprocal of N.
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _input = input + bid * N;
    const bf16_t* _residual = residual + bid * N;
    fp8_e4m3_t* _output = output + bid * N;

    fp32_t* _scales;
     _scales = scales + bid;

    // Shared memory workspace to store vectorized (half2) data.
    // Note: since each bf16x2_t holds 2 half values, the workspace size is N/2.
    extern __shared__ bf16x2_t workspace2[];

    // Local registers to hold vectorized data.
    bf16x2_t local_input[VPT / 2];
    bf16x2_t local_residual[VPT / 2];
    bf16x2_t local_w[VPT / 2];
    bf16x2_t local_output[VPT / 2];
    fp8x4_e4m3_t local_f8[VPT / 4];
    

    // Each thread computes a partial sum of squares.
    fp32_t local_square_sum = 0.0f;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT FP16 elements from global memory (_input) into local vector (local_input).
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_input);
        // Load VPT FP16 elements from global memory (_residual) into local vector (local_residual).
        vec_copy<sizeof(bf16_t) * VPT>(_residual + i, local_residual);

        # pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            // Convert the bf16x2_t to fp32x2_t for computation.
            fp32x2_t x = bf16x2_to_fp32x2(local_input[j]);
            fp32x2_t r = bf16x2_to_fp32x2(local_residual[j]);
            // Add the residual to the input.
            local_input[j] = _float22bf162_rn(make_float2(x.x + r.x, x.y + r.y));

            fp32x2_t tmp = bf16x2_to_fp32x2(local_input[j]);
            local_square_sum += (tmp.x * tmp.x + tmp.y * tmp.y);
        }

        // Store the loaded data into shared memory.
        // Divide index by 2 because 'workspace' is an array of bf16x2_t.
        vec_copy<sizeof(bf16_t) * VPT>(local_input, _input + i);
        vec_copy<sizeof(bf16_t) * VPT>(local_input, workspace2 + (i >> 1));
    }

    const fp32_t reduced_square_sum = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum); 

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    const fp32_t mean_square = reduced_square_sum * r_N;
    const fp32_t inv_norm = rsqrtf(mean_square + eps);

    // Normalize each element using the computed normalization factor.
    fp32_t local_max = -FLT_MAX;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load the previously stored vectorized data from shared memory.
        vec_copy<sizeof(bf16_t) * VPT>(workspace2 + (i >> 1), local_input);
        // Load the corresponding weight values from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(weight + i, local_w);

        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_input[j]);
            fp32x2_t w = bf16x2_to_fp32x2(local_w[j]);
            // Apply normalization: multiply by inv_norm and then scale by the weight.
            fp32x2_t ret = make_float2(
                x.x * inv_norm * w.x,
                x.y * inv_norm * w.y
            );
            local_output[j] = _float22bf162_rn(ret);


            fp32x2_t tmp = bf16x2_to_fp32x2(local_output[j]);
            fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
            local_max = fmaxf(local_max, max);
        }

        vec_copy<sizeof(bf16_t) * VPT>(local_output, workspace2 + (i >> 1));
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace2 + (i >> 1), local_output);

        #pragma unroll
        for (int32_t j = 0; j < VPT/4; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_output[2 * j + 0]);
            fp32x2_t y = bf16x2_to_fp32x2(local_output[2 * j + 1]);
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


template<int32_t TPB, int32_t N>
__global__ void device_add_norm_quant_bf16(
    bf16_t* __restrict__ input,  // Input tensor in BF16 format
    const bf16_t* __restrict__ residual, // Residual tensor in BF16 format
    const bf16_t* __restrict__ weight, // Weight tensor in BF16 format
    fp8_e4m3_t* __restrict__ output,   // Output tensor in FP8 format
    fp32_t* __restrict__ scales,       // Output scales for each group
    const int64_t M,                   // Number of rows in the input tensor
    const fp32_t eps                   // Epsilon value for numerical stability
) {
    constexpr int32_t VPT = 8;                // Number of FP16 values processed per thread.
    constexpr fp32_t r_N = 1 / (fp32_t)N;       // Reciprocal of N.
    constexpr fp32_t FP8_E4M3_MAX = 448.0f; // Maximum value representable in FP8 E4M3 format

    static_assert(N % 2 == 0, "N must be even.");
    static_assert(N % VPT == 0, "N must be a multiple of VPT.");

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _input = input + bid * N;
    const bf16_t* _residual = residual + bid * N;
    fp8_e4m3_t* _output = output + bid * N;

    fp32_t* _scales;
     _scales = scales + bid;

    // Shared memory workspace to store vectorized (half2) data.
    // Note: since each bf16x2_t holds 2 half values, the workspace size is N/2.
    __shared__ bf16x2_t workspace[N / 2];

    // Local registers to hold vectorized data.
    bf16x2_t local_input[VPT / 2];
    bf16x2_t local_residual[VPT / 2];
    bf16x2_t local_w[VPT / 2];
    bf16x2_t local_output[VPT / 2];
    fp8x4_e4m3_t local_f8[VPT / 4];
    

    // Each thread computes a partial sum of squares.
    fp32_t local_square_sum = 0.0f;
    # pragma unroll
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT FP16 elements from global memory (_input) into local vector (local_input).
        vec_copy<sizeof(bf16_t) * VPT>(_input + i, local_input);
        // Load VPT FP16 elements from global memory (_residual) into local vector (local_residual).
        vec_copy<sizeof(bf16_t) * VPT>(_residual + i, local_residual);

        # pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            // Convert the bf16x2_t to fp32x2_t for computation.
            fp32x2_t x = bf16x2_to_fp32x2(local_input[j]);
            fp32x2_t r = bf16x2_to_fp32x2(local_residual[j]);
            // Add the residual to the input.
            local_input[j] = _float22bf162_rn(make_float2(x.x + r.x, x.y + r.y));

            fp32x2_t tmp = bf16x2_to_fp32x2(local_input[j]);
            local_square_sum += (tmp.x * tmp.x + tmp.y * tmp.y);
        }

        // Store the loaded data into shared memory.
        // Divide index by 2 because 'workspace' is an array of bf16x2_t.
        vec_copy<sizeof(bf16_t) * VPT>(local_input, _input + i);
        vec_copy<sizeof(bf16_t) * VPT>(local_input, workspace + (i >> 1));
    }

    const fp32_t reduced_square_sum = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum); 

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    const fp32_t mean_square = reduced_square_sum * r_N;
    const fp32_t inv_norm = rsqrtf(mean_square + eps);

    // Normalize each element using the computed normalization factor.
    fp32_t local_max = -FLT_MAX;
    #pragma unroll
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load the previously stored vectorized data from shared memory.
        vec_copy<sizeof(bf16_t) * VPT>(workspace + (i >> 1), local_input);
        // Load the corresponding weight values from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(weight + i, local_w);

        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_input[j]);
            fp32x2_t w = bf16x2_to_fp32x2(local_w[j]);
            // Apply normalization: multiply by inv_norm and then scale by the weight.
            fp32x2_t ret = make_float2(
                x.x * inv_norm * w.x,
                x.y * inv_norm * w.y
            );
            local_output[j] = _float22bf162_rn(ret);


            fp32x2_t tmp = bf16x2_to_fp32x2(local_output[j]);
            fp32_t max = fmaxf(fabsf(tmp.x), fabsf(tmp.y));
            local_max = fmaxf(local_max, max);
        }

        vec_copy<sizeof(bf16_t) * VPT>(local_output, workspace + (i >> 1));
    }

    // Reduce the maximum value across the block
    const fp32_t reduced_max = lightllm::reduce::sm70::sync_block_reduce_max_f32<TPB>(local_max);

    // Compute the scale factor with epsilon to avoid division by zero
    constexpr fp32_t epsilon = 1e-7f;
    const fp32_t scale = reduced_max / FP8_E4M3_MAX;
    const fp32_t inv_scale = 1.0f / (scale + epsilon);

    #pragma unroll
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        vec_copy<sizeof(bf16_t) * VPT>(workspace + (i >> 1), local_output);

        #pragma unroll
        for (int32_t j = 0; j < VPT/4; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_output[2 * j + 0]);
            fp32x2_t y = bf16x2_to_fp32x2(local_output[2 * j + 1]);
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

/**
 * @brief Fused add norm quant
 */
std::tuple<Tensor, Tensor> add_norm_quant_bf16_fp8(
    Tensor& X, const Tensor &R, const Tensor &W,
    const fp32_t eps
) {
    TORCH_CHECK(X.ndimension() == 2, "Input tensor X must be 2D");
    TORCH_CHECK(R.ndimension() == 2, "Input tensor R must be 2D");
    TORCH_CHECK(W.ndimension() == 1, "Input tensor W must be 1D");

    TORCH_CHECK(X.is_cuda(), "Input tensor X must be a CUDA tensor.");
    TORCH_CHECK(R.is_cuda(), "Input tensor R must be a CUDA tensor.");
    TORCH_CHECK(W.is_cuda(), "Input tensor W must be a CUDA tensor.");

    TORCH_CHECK(X.scalar_type() == c10::ScalarType::BFloat16, "Input tensor X must be BF16.");
    TORCH_CHECK(R.scalar_type() == c10::ScalarType::BFloat16, "Input tensor R must be BF16.");
    TORCH_CHECK(W.scalar_type() == c10::ScalarType::BFloat16, "Input tensor W must be BF16.");

    Tensor contiguous_X = X.is_contiguous() ? X : X.contiguous();
    Tensor contiguous_R = R.is_contiguous() ? R : R.contiguous();
    Tensor contiguous_W = W.is_contiguous() ? W : W.contiguous();

    const uint32_t M = contiguous_X.size(0);
    const uint32_t N = contiguous_X.size(1);
    
    Tensor output_q = torch::empty(
        {M, N},
        torch::TensorOptions()
            .dtype(torch::kFloat8_e4m3fn)
            .device(contiguous_X.device())
    );
    Tensor scales = torch::empty(
        {M, 1},
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(contiguous_X.device())
    );

    const int32_t blocks = M;

    switch (N) {
        case 16:
            device_add_norm_quant_bf16<128, 16>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 32:
            device_add_norm_quant_bf16<128, 32>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 64:
            device_add_norm_quant_bf16<128, 64>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 512:
            device_add_norm_quant_bf16<128, 512>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 1024:
            device_add_norm_quant_bf16<128, 1024>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 3200:
            device_add_norm_quant_bf16<128, 3200>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 4096:
            device_add_norm_quant_bf16<128, 4096>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        case 12800:
            device_add_norm_quant_bf16<256, 12800>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(contiguous_X),
                PTR<bf16_t>(contiguous_R),
                PTR<bf16_t>(contiguous_W),
                PTR<fp8_e4m3_t>(output_q),
                PTR<fp32_t>(scales),
                M,
                eps
            );
            break;
        default: {
            static constexpr int32_t TPB = 128;
            const int64_t shared_mem_size = N * sizeof(bf16_t);
            if (N % 8 == 0) {
                device_add_norm_quant_bf16_vpt<TPB>
                <<<blocks, TPB, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_X),
                    PTR<bf16_t>(contiguous_R),
                    PTR<bf16_t>(contiguous_W),
                    PTR<fp8_e4m3_t>(output_q),
                    PTR<fp32_t>(scales),
                    M,
                    N,
                    eps
                );
            } else {
                device_add_norm_quant_bf16_general<TPB>
                <<<blocks, TPB, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(contiguous_X),
                    PTR<bf16_t>(contiguous_R),
                    PTR<bf16_t>(contiguous_W),
                    PTR<fp8_e4m3_t>(output_q),
                    PTR<fp32_t>(scales),
                    M,
                    N,
                    eps
                );
            }
        }
    }

    return {output_q, scales};
}

} // namespace ops
} // namespace lightllm