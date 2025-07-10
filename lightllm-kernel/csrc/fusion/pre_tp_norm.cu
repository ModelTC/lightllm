#include "ops_common.h"
#include "reduce/sm70.cuh"

namespace lightllm {
namespace ops {

using namespace lightllm;

/**
 * @tparam TPB   Threads per block.
 * @tparam N     Number of bf16 elements in one row.
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 */
template<int32_t TPB>
__global__
void device_pre_tp_norm_bf16_general(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    fp32_t __restrict__ *V,                        // [M] Variance tensor pointer.
    const int32_t M,                  // Number of rows.
    const int32_t N
) {
    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;

    bf16_t local_x = cvt_f32_bf16(0.0f);
    fp32_t local_square_sum = 0.0f;
    for (int32_t i = tid; i < N; i += TPB) {
        local_x = _X[i];

        fp32_t tmp = cvt_bf16_f32(local_x);

        local_square_sum += tmp * tmp;
    }

    fp32_t block_square_sum = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum);

    if (tid == 0) {
        V[bid] = block_square_sum;
    }

}



/**
 * @tparam TPB   Threads per block.
 * @tparam N     Number of bf16 elements in one row (must be a multiple of VPT).
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 */
template<int32_t TPB>
__global__
void device_pre_tp_norm_bf16_vpt(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    fp32_t __restrict__ *V,                        // [M] Variance tensor pointer.
    const int32_t M,                  // Number of rows.
    const int32_t N
) {
    constexpr int32_t VPT = 8;                // Number of bf16 values processed per thread.

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;

    // Local registers to hold vectorized data.
    bf16x2_t local_x[VPT / 2];

    // Each thread computes a partial sum of squares.
    fp32_t local_square_sum = 0.0f;
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT bf16 elements from global memory (_X) into local vector (local_x).
        vec_copy<sizeof(bf16_t) * VPT>(_X + i, local_x);

        // Compute the sum of squares for the VPT elements.
        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t tmp = bf16x2_to_fp32x2(local_x[j]);
            local_square_sum += (tmp.x * tmp.x + tmp.y * tmp.y);
        }
    }

    // Reduce the partial sums across the block, block reduce sum will invoke __syncthread();
    V[bid] = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum);

}


/**
 * @tparam TPB   Threads per block.
 * @tparam N     Number of bf16 elements in one row (must be a multiple of VPT).
 *
 * @param X       Pointer to the input tensor in global memory. [M, N]
 * @param M       Number of rows in the tensor.
 */
template<int32_t TPB, int32_t N>
__global__
void device_pre_tp_norm_bf16(
    bf16_t __restrict__ *X,           // [M, N] Input tensor pointer.
    fp32_t __restrict__ *V,                        // [M] Variance tensor pointer.
    const int32_t M                  // Number of rows.
) {
    constexpr int32_t VPT = 8;                // Number of bf16 values processed per thread.

    static_assert(N % 2 == 0, "N must be even.");
    static_assert(N % VPT == 0, "N must be a multiple of VPT.");

    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;

    // Each block processes one row of the input tensor.
    bf16_t* _X = X + bid * N;

    // Local registers to hold vectorized data.
    bf16x2_t local_x[VPT / 2];

    // Each thread computes a partial sum of squares.
    fp32_t local_square_sum = 0.0f;
    # pragma unroll
    for (int32_t i = tid * VPT; i < N; i += TPB * VPT) {
        // Load VPT bf16 elements from global memory (_X) into local vector (local_x).
        vec_copy<sizeof(bf16_t) * VPT>(_X + i, local_x);

        // Compute the sum of squares for the VPT elements.
        #pragma unroll
        for (int32_t j = 0; j < VPT / 2; j++) {
            fp32x2_t tmp = bf16x2_to_fp32x2(local_x[j]);
            local_square_sum += (tmp.x * tmp.x + tmp.y * tmp.y);
        }
    }

    // Reduce the partial sums across the block, block reduce sum will invoke __syncthread();
    V[bid] = lightllm::reduce::sm70::sync_block_reduce_sum_f32<TPB>(local_square_sum);

}

/**
 * @param X    Input tensor with shape [M, N] (bf16, CUDA).
 */
Tensor pre_tp_norm_bf16(Tensor &X) {
    TORCH_CHECK(X.ndimension() == 2 || X.ndimension() == 4, "Input tensor must be 2D or 4D");
    TORCH_CHECK(X.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(X.scalar_type() == c10::ScalarType::BFloat16, "Input tensor must be BF16.");

    Tensor contiguous_X = X.is_contiguous() ? X : X.contiguous();
    Tensor input_tensor;
    uint32_t M, N;
    Tensor V;

    if (X.ndimension() == 2) {
        M = contiguous_X.size(0);
        N = contiguous_X.size(1);
        input_tensor = contiguous_X;
        V = torch::empty(
            {M},
            torch::TensorOptions()
                .dtype(c10::ScalarType::Float)
                .device(contiguous_X.device())
        );
    } else {
        const uint32_t d0 = contiguous_X.size(0);
        const uint32_t d1 = contiguous_X.size(1);
        const uint32_t d2 = contiguous_X.size(2);
        const uint32_t d3 = contiguous_X.size(3);

        M = d0 * d1;
        N = d2 * d3;
        input_tensor = contiguous_X.view({M, N});
        V = torch::empty(
            {M},
            torch::TensorOptions()
                .dtype(c10::ScalarType::Float)
                .device(contiguous_X.device())
        );
    }


    // Each CUDA block processes one row.
    const int32_t blocks = M;

    // Kernel dispatch based on the value of N.
    switch (N) {
        case 768:
            device_pre_tp_norm_bf16<128, 768>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 1024:
            device_pre_tp_norm_bf16<128, 1024>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 1664:
            device_pre_tp_norm_bf16<128, 1664>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 2048:
            device_pre_tp_norm_bf16<128, 2048>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 3200:
            device_pre_tp_norm_bf16<128, 3200>
            <<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 4096:
            device_pre_tp_norm_bf16<256, 4096>
            <<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 8192:
            device_pre_tp_norm_bf16<512, 8192>
            <<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        case 10240:
            device_pre_tp_norm_bf16<512, 10240>
            <<<blocks, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
                PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M
            );
            break;
        default: {
            static constexpr int32_t TPB = 256;
            if (N % 8 == 0) {
                device_pre_tp_norm_bf16_vpt<TPB>
                <<<blocks, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M, N
                );
            } else {
                device_pre_tp_norm_bf16_general<TPB>
                <<<blocks, TPB, 0, at::cuda::getCurrentCUDAStream()>>>(
                    PTR<bf16_t>(input_tensor), PTR<fp32_t>(V), M, N
                );
            }
        }
    }
    return V;
}

} // namespace ops
} // namespace lightllm