#include "ops_common.h"
#include "reduce/sm70.cuh"

namespace lightllm {
namespace ops {

using namespace lightllm;

/**
 * @brief CUDA kernel to perform RMS normalization on an FP16 tensor.
 *
 * Each block processes one row of the input tensor. 
 *
 * @tparam TPB   Threads per block.
 * @tparam N     Number of FP16 elements in one row.
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param W       Pointer to the weight tensor in global memory. [N]
 * @param V       Pointer to the variance tensor in global memory. [M]
 * @param Y       Pointer to the output tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 * @param eps     Epsilon for numerical stability.
 */
template<int32_t TPB>
__global__
void  device_post_tp_norm_bf16_general(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    const bf16_t __restrict__ *W,     // [N] Weight tensor pointer.
    const fp32_t __restrict__ *V,     // [M] variance
    bf16_t __restrict__ *Y,                        // [M, N] Output tensor pointer.
    const int32_t M,                  // Number of rows.
    const int32_t N,
    const int32_t embed_dim,          // if multiGPUs, embed_dim differs from N
    const fp32_t eps                  // Epsilon for numerical stability.
) {
    const fp32_t r_N = 1 / (fp32_t)embed_dim;       // Reciprocal of N.

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;
    bf16_t* _Y = Y + bid * N;

    // Local registers to hold data.
    bf16_t local_x = cvt_f32_bf16(0.0f);
    bf16_t local_w = cvt_f32_bf16(0.0f);
    bf16_t local_y = cvt_f32_bf16(0.0f);

    fp32_t reduced_square_sum = V[bid];

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    fp32_t mean_square = reduced_square_sum * r_N;
    fp32_t inv_norm = rsqrtf(mean_square + eps);

    for (int32_t i = tid; i < N; i += TPB) {
        local_x = _X[i];
        local_w = W[i];

        fp32_t x = cvt_bf16_f32(local_x);
        fp32_t w = cvt_bf16_f32(local_w);

        fp32_t ret = x * inv_norm * w;
        local_y = cvt_f32_bf16(ret);

        _Y[i] = local_y;
    }
}


/**
 * @brief CUDA kernel to perform RMS normalization on an FP16 tensor.
 *
 * Each block processes one row of the input tensor. The kernel loads the
 * data in a vectorized manner (using half2), computes the mean square,
 * calculates the reciprocal square root (i.e. 1/sqrt(mean_square+eps)),
 * and then normalizes the input row element‐wise while scaling with a weight.
 *
 * @tparam TPB   Threads per block.
 * @tparam N     Number of FP16 elements in one row (must be a multiple of VPT).
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param W       Pointer to the weight tensor in global memory. [N]
 * @param V       Pointer to the variance tensor in global memory. [M]
 * @param Y       Pointer to the output tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 * @param eps     Epsilon for numerical stability.
 */
template<int32_t TPB>
__global__
void  device_post_tp_norm_bf16_vpt(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    const bf16_t __restrict__ *W,     // [N] Weight tensor pointer.
    const fp32_t __restrict__ *V,     // [M] variance
    bf16_t __restrict__ *Y,                        // [M, N] Output tensor pointer.
    const int32_t M,                  // Number of rows.
    const int32_t N,
    const int32_t embed_dim,          // if multiGPUs, embed_dim differs from N
    const fp32_t eps                  // Epsilon for numerical stability.
) {
    constexpr int32_t VPT = 8;                // Number of bf16 values processed per thread.
    const fp32_t r_N = 1 / (fp32_t)embed_dim;       // Reciprocal of N.

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;
    bf16_t* _Y = Y + bid * N;

    // Local registers to hold vectorized data.
    bf16x2_t local_x[VPT / 2];
    bf16x2_t local_w[VPT / 2];
    bf16x2_t local_y[VPT / 2];

    fp32_t reduced_square_sum = V[bid];

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    fp32_t mean_square = reduced_square_sum * r_N;
    fp32_t inv_norm = rsqrtf(mean_square + eps);

    // Normalize each element using the computed normalization factor.
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load the previously stored vectorized data from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(_X + i, local_x);
        // Load the corresponding weight values from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(W + i, local_w);

        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_x[j]);
            fp32x2_t w = bf16x2_to_fp32x2(local_w[j]);
            // Apply normalization: multiply by inv_norm and then scale by the weight.
            fp32x2_t ret = make_float2(
                x.x * inv_norm * w.x,
                x.y * inv_norm * w.y
            );
            local_y[j] = _float22bf162_rn(ret);
        }
        // Write the normalized vectorized data back to global memory.
        vec_copy<sizeof(bf16_t) * VPT>(local_y, _Y + i);
    }
}

/**
 * @brief CUDA kernel to perform RMS normalization on an FP16 tensor.
 *
 * Each block processes one row of the input tensor. The kernel loads the
 * data in a vectorized manner (using half2), computes the mean square,
 * calculates the reciprocal square root (i.e. 1/sqrt(mean_square+eps)),
 * and then normalizes the input row element‐wise while scaling with a weight.
 *
 * @tparam TPB   Threads per block.
 * @tparam N     Number of FP16 elements in one row (must be a multiple of VPT).
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param W       Pointer to the weight tensor in global memory. [N]
 * @param V       Pointer to the variance tensor in global memory. [M]
 * @param Y       Pointer to the output tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 * @param eps     Epsilon for numerical stability.
 */
template<int32_t TPB, int32_t N>
__global__
void  device_post_tp_norm_bf16(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    const bf16_t __restrict__ *W,     // [N] Weight tensor pointer.
    const fp32_t __restrict__ *V,     // [M] variance
    bf16_t __restrict__ *Y,                        // [M, N] Output tensor pointer.
    const int32_t M,                  // Number of rows.
    const int32_t embed_dim,          // if multiGPUs, embed_dim differs from N
    const fp32_t eps                  // Epsilon for numerical stability.
) {
    constexpr int32_t VPT = 8;                // Number of bf16 values processed per thread.
    const fp32_t r_N = 1 / (fp32_t)embed_dim;       // Reciprocal of N.

    static_assert(N % 2 == 0, "N must be even.");
    static_assert(N % VPT == 0, "N must be a multiple of VPT.");

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;
    bf16_t* _Y = Y + bid * N;

    // Local registers to hold vectorized data.
    bf16x2_t local_x[VPT / 2];
    bf16x2_t local_w[VPT / 2];
    bf16x2_t local_y[VPT / 2];

    fp32_t reduced_square_sum = V[bid];

    // Compute the mean square and then the inverse RMS normalization factor.
    // For RMSNorm, the normalization factor is 1/sqrt(mean(x^2)+eps).
    fp32_t mean_square = reduced_square_sum * r_N;
    fp32_t inv_norm = rsqrtf(mean_square + eps);

    // Normalize each element using the computed normalization factor.
    # pragma unroll
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load the previously stored vectorized data from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(_X + i, local_x);
        // Load the corresponding weight values from global memory.
        vec_copy<sizeof(bf16_t) * VPT>(W + i, local_w);

        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t x = bf16x2_to_fp32x2(local_x[j]);
            fp32x2_t w = bf16x2_to_fp32x2(local_w[j]);
            // Apply normalization: multiply by inv_norm and then scale by the weight.
            fp32x2_t ret = make_float2(
                x.x * inv_norm * w.x,
                x.y * inv_norm * w.y
            );
            local_y[j] = _float22bf162_rn(ret);
        }
        // Write the normalized vectorized data back to global memory.
        vec_copy<sizeof(bf16_t) * VPT>(local_y, _Y + i);
    }
}

/**
 * @brief Launch RMSNorm kernel for FP16 tensors with aligned 16-element rows.
 *
 * This function validates the input tensors, ensures they are contiguous,
 * selects the appropriate kernel configuration based on the row width N,
 * and launches the CUDA kernel.
 *
 * @param X    Input tensor with shape [M, N] (FP16, CUDA).
 * @param W    Weight tensor with shape [N] (FP16, CUDA).
 * @param eps  Epsilon for numerical stability.
 * @return     Output tensor with the same shape as X.
 */
Tensor post_tp_norm_bf16(Tensor &X, const Tensor &W, const Tensor &V, const int embed_dim, const fp32_t eps) {
    TORCH_CHECK(X.ndimension() == 2 || X.ndimension() == 4, "Input tensor must be 2D or 4D");
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(X.scalar_type() == c10::ScalarType::BFloat16, "Input tensor must be BF16.");

    Tensor contiguous_X = X.is_contiguous() ? X : X.contiguous();
    Tensor contiguous_W = W.is_contiguous() ? W : W.contiguous();
    Tensor contiguous_V = V.is_contiguous() ? V : V.contiguous();

    Tensor input_tensor;
    uint32_t M, N;
    Tensor Y;

    if (X.ndimension() == 2) {
        M = contiguous_X.size(0);
        N = contiguous_X.size(1);
        input_tensor = contiguous_X;
        Y = torch::empty_like(input_tensor);
    } else {
        const uint32_t d0 = contiguous_X.size(0);
        const uint32_t d1 = contiguous_X.size(1);
        const uint32_t d2 = contiguous_X.size(2);
        const uint32_t d3 = contiguous_X.size(3);

        M = d0 * d1;
        N = d2 * d3;
        input_tensor = contiguous_X.view({M, N});
        Y = torch::empty_like(input_tensor);
    }

    // Each CUDA block processes one row.
    const int32_t blocks = M;

    // Kernel dispatch based on the value of N.
    switch (N) {
        case 768:
            device_post_tp_norm_bf16<128, 768>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 1024:
            device_post_tp_norm_bf16<128, 1024>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 1664:
            device_post_tp_norm_bf16<128, 1664>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 2048:
            device_post_tp_norm_bf16<128, 2048>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 3200:
            device_post_tp_norm_bf16<128, 3200>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
        break;
        case 4096:
            device_post_tp_norm_bf16<256, 4096>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 8192:
            device_post_tp_norm_bf16<512, 8192>
            <<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        case 10240:
            device_post_tp_norm_bf16<512, 10240>
            <<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                M, embed_dim, eps
            );
            break;
        default:
            static constexpr int32_t TPB = 256;
            if (N % 8 == 0) {
                device_post_tp_norm_bf16_vpt<TPB>
                <<<blocks, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                    PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                    M, N, embed_dim, eps
                );
            } else {
                device_post_tp_norm_bf16_general<TPB>
                <<<blocks, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(input_tensor), PTR<bf16_t>(contiguous_W),
                    PTR<fp32_t>(contiguous_V), PTR<bf16_t>(Y),
                    M, N, embed_dim, eps
                );
            }
    }

    // need to reshape Y back to 4 dimens
    if (X.ndimension() == 4) {
        Y = Y.reshape(X.sizes());
    }

    return Y;
}

} // namespace ops
} // namespace lightllm