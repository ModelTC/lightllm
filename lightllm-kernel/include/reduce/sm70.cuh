#pragma once
#include "utils.h"

namespace lightllm {
namespace reduce {
namespace sm70 {
/**
 * @brief Performs a block-wide reduction to sum up floating-point
 * values across all threads in a block.
 *
 * This function computes the sum of all `input` values
 * provided by threads in a block using
 * a combination of warp shuffle and shared memory.
 * The result is stored in the first thread of the block.
 *
 * @tparam TPB Threads per block, must be a multiple of the warp size (32).
 * @param input The input value for the calling thread.
 * @return The block-wide sum of the input values. Only thread 0 of the block holds the valid result.
 *
 * @note This function assumes that `TPB` is divisible by 32 (warp size).
 */
template<int32_t TPB>
__device__ inline
fp32_t sync_block_reduce_sum_f32(const fp32_t input) {
    constexpr int32_t warpSize = 32;
    static_assert(TPB <= warpSize * warpSize);

    // Thread ID within the current block
    const int32_t tid = threadIdx.x;
    const int32_t warp_lane = tid % 32;
    const int32_t warp_id   = tid / warpSize;

    fp32_t local_sum = input;

    // Warp-level reduction using shuffle operations
    for (int32_t stride = warpSize / 2; stride > 0; stride /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, stride);
    }

    // Shared memory reduction across warps
    __shared__ fp32_t shared_sum[TPB / warpSize];
    if (warp_lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Block-level reduction using the first warp
    if (warp_id == 0) {
        if (warp_lane < TPB / warpSize) {
            local_sum = shared_sum[warp_lane];
        } else {
            local_sum = 0.0f;
        }

        for (int32_t stride = (TPB / warpSize) / 2; stride > 0; stride /= 2) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, stride);
        }
    }

    if (warp_id == 0 && warp_lane == 0) {
        shared_sum[0] = local_sum;
    }
    __syncthreads();

    return shared_sum[0];
}



template<int32_t TPB>
__device__ inline
fp32_t sync_block_reduce_max_f32(const fp32_t input) {
    constexpr int32_t warpSize = 32;
    static_assert(TPB <= warpSize * warpSize);

    // Thread ID within the current block
    const int32_t tid = threadIdx.x;
    const int32_t warp_lane = tid % 32;
    const int32_t warp_id   = tid / warpSize;

    fp32_t local_max = input;

    // Warp-level reduction using shuffle operations
    for (int32_t stride = warpSize / 2; stride > 0; stride /= 2) {
        local_max = fmaxf(__shfl_down_sync(0xFFFFFFFF, local_max, stride), local_max);
    }

    // Shared memory reduction across warps
    __shared__ fp32_t shared_max[TPB / warpSize];
    if (warp_lane == 0) {
        shared_max[warp_id] = local_max;
    }
    __syncthreads();

    // Block-level reduction using the first warp
    if (warp_id == 0) {
        if (warp_lane < TPB / warpSize) {
            local_max = shared_max[warp_lane];
        } else {
            local_max = -FLT_MAX;
        }

        for (int32_t stride = (TPB / warpSize) / 2; stride > 0; stride /= 2) {
            local_max = fmaxf(__shfl_down_sync(0xFFFFFFFF, local_max, stride), local_max);
        }
    }

    if (warp_id == 0 && warp_lane == 0) {
        shared_max[0] = local_max;
    }
    __syncthreads();

    return shared_max[0];
}

/**
 * @brief Performs a block-wide reduction to compute both sum and max
 * of floating-point values across all threads in a block.
 *
 * This function computes both the sum and maximum of all `input` values
 * provided by threads in a block using a combination of warp shuffle
 * and shared memory. The result is stored in the first thread of the block.
 *
 * @tparam TPB Threads per block, must be a multiple of the warp size (32).
 * @param input The input value for the calling thread (contains .x for sum, .y for max).
 * @return The block-wide reduction result (sum in .x, max in .y). Only thread 0 of the block holds the valid result.
 *
 * @note This function assumes that `TPB` is divisible by 32 (warp size).
 */
template<int32_t TPB>
__device__ inline
fp32x2_t sync_block_reduce_sum_max_f32(const fp32x2_t input) {
    constexpr int32_t warpSize = 32;
    static_assert(TPB <= warpSize * warpSize);

    // Thread ID within the current block
    const int32_t tid = threadIdx.x;
    const int32_t warp_lane = tid % warpSize;
    const int32_t warp_id   = tid / warpSize;

    fp32x2_t local_result = input;

    // Warp-level reduction using shuffle operations
    for (int32_t stride = warpSize / 2; stride > 0; stride /= 2) {
        // Sum reduction for .x component
        float sum_val = __shfl_down_sync(0xFFFFFFFF, local_result.x, stride);
        local_result.x += sum_val;
        
        // Max reduction for .y component
        float max_val = __shfl_down_sync(0xFFFFFFFF, local_result.y, stride);
        local_result.y = max(local_result.y, max_val);
    }

    // Shared memory reduction across warps
    __shared__ fp32x2_t shared_result[TPB / warpSize];
    if (warp_lane == 0) {
        shared_result[warp_id] = local_result;
    }
    __syncthreads();

    // Block-level reduction using the first warp
    if (warp_id == 0) {
        if (warp_lane < TPB / warpSize) {
            local_result = shared_result[warp_lane];
        } else {
            local_result.x = 0.0f;  // Identity for sum
            local_result.y = -INFINITY;  // Identity for max
        }

        for (int32_t stride = (TPB / warpSize) / 2; stride > 0; stride /= 2) {
            // Sum reduction for .x component
            float sum_val = __shfl_down_sync(0xFFFFFFFF, local_result.x, stride);
            local_result.x += sum_val;
            
            // Max reduction for .y component
            float max_val = __shfl_down_sync(0xFFFFFFFF, local_result.y, stride);
            local_result.y = max(local_result.y, max_val);
        }
    }

    if (warp_id == 0 && warp_lane == 0) {
        shared_result[0] = local_result;
    }
    __syncthreads();

    return shared_result[0];
}

} // namespace sm70
} // namespace reduce
} // namespace lightllm