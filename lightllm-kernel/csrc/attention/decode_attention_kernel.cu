#include <cuda_fp16.h>
#include <float.h> // need for FLT_MAX
#include <math.h>
#include <memory>
#include <assert.h>
#include "ops_common.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace lightllm {
namespace ops {

# include <torch/extension.h>
#define LIGHT_DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define LIGHT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)             \
  AT_DISPATCH_SWITCH(                                             \
    TYPE, NAME, LIGHT_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

template <typename T>
__device__ inline float tofloat(T value) {
    return static_cast<float>(value);
}

// Specialization for __half
template <>
__device__ inline float tofloat<__half>(__half value) {
    return __half2float(value);
}

// Specialization for __nv_bfloat16
template <>
__device__ inline float tofloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<int32_t THREAD_GROUP_SIZE, int32_t ELEMENT_NUM, typename T>
__device__ inline
float attn_thread_group_dot(T* local_q, T* local_k)
{
    // Helper function for QK Dot.
    // [TODO] It should be optimized by type fp32x4.

    float qk = 0.0f;
# pragma unroll
    for(int32_t i = 0; i < ELEMENT_NUM; i++) {
        qk += tofloat(local_q[i]) * tofloat(local_k[i]);
    }
#pragma unroll
    for (int32_t mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
    }
    return qk;
}

template<int32_t WPT>
__device__ inline
float attn_block_reduce_max(float reducing, float* shared_mem)
{
    // Helper function for reduce softmax qkmax.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    if (lane_id == 0) {
        shared_mem[warp_id] = reducing;
    }
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];
    else reducing = -FLT_MAX;

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing = fmaxf(reducing, __shfl_xor_sync(uint32_t(-1), reducing, mask));
    }

    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<int32_t WPT>
__device__ inline
float attn_block_reduce_sum(float reducing, float *shared_mem)
{
    // Helper function for reduce softmax exp sum.
    constexpr int32_t WARP_SIZE = 32;
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t warp_id = threadIdx.x / WARP_SIZE;

# pragma unroll
    for (int32_t mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }

    if (lane_id == 0) shared_mem[warp_id] = reducing;
    __syncthreads();

    if (lane_id < WPT) reducing = shared_mem[lane_id];

# pragma unroll
    for (int32_t mask = WPT / 2; mask >= 1; mask /= 2) {
        reducing += __shfl_xor_sync(uint32_t(-1), reducing, mask);
    }
    reducing = __shfl_sync(uint32_t(-1), reducing, 0);
    return reducing;
}

template<
    int32_t HEAD_SIZE,
    int32_t THREAD_GROUP_SIZE,        // how many threads inside a group
    int32_t TPB,
    int32_t QUANT_GROUP,
    typename T>
__global__
void dynamic_batching_decoding_cache_attention_fp16_kernel(
    T* __restrict__ output,          // [context_lens, num_heads..., head_size]
    
    const T* __restrict__ query,     // [seq_lens, num_heads..., head_size]
    const int8_t* k_cache,                // [max_token, num_kv_heads, head_size]
    const T* k_scale,                  // [max_token, num_kv_heads, head_size / quant_group(8)]
    const int8_t* v_cache,                // [max_token, num_kv_heads, head_size]
    const T* v_scale,                  // [max_token, num_kv_heads, head_size / quant_group(8)]

    const float attn_scale,

    const int64_t output_stride_s,
    const int64_t output_stride_h,

    const int64_t query_stride_s,
    const int64_t query_stride_h,

    const int64_t kcache_stride_s,
    const int64_t kcache_stride_h,

    const int64_t vcache_stride_s,
    const int64_t vcache_stride_h,

    const int32_t * __restrict__ b_seq_len,
    const int32_t * __restrict__ b_req_idx,
    const int32_t * __restrict__ req_to_tokens,
    const int64_t req_to_tokens_stride,
    const int64_t max_len_in_batch,
    const int64_t gqa_group_size) { 

    /* --- Decoding Attention Kernel Implementation --- */
    constexpr int64_t WARP_SIZE = 32;                              // warp size
    constexpr int64_t WPT       = TPB / WARP_SIZE;                 // warp per thread block， TPB for Thread per block 4, block_size
    constexpr int64_t GPW       = WARP_SIZE / THREAD_GROUP_SIZE;       // thread group per warp 4
    constexpr int64_t GPT       = WARP_SIZE / THREAD_GROUP_SIZE * WPT; // thread group per thread block 16

    // const int64_t num_heads     = gridDim.x;
    const int64_t head_idx      = blockIdx.x;
    const int64_t batch_idx     = blockIdx.y;

    const int64_t seq_len = b_seq_len[batch_idx];
    const int64_t cur_req_idx = b_req_idx[batch_idx];
    const int32_t * b_start_loc = req_to_tokens + cur_req_idx * req_to_tokens_stride;

    constexpr int64_t VEC_SIZE  = 16 / sizeof(T);  // 128 bits, 这个是 cuda 能操作的最大的一个单位的数吧，8

    // ------------------------------------------------ //
    // Step 1. Load Q into Thread Reg.
    constexpr int64_t VEC_LEN = (HEAD_SIZE / VEC_SIZE) / THREAD_GROUP_SIZE; // 128 / 8 / 8 = 2

    static_assert((HEAD_SIZE / THREAD_GROUP_SIZE) % VEC_SIZE == 0);
    static_assert(HEAD_SIZE % THREAD_GROUP_SIZE == 0);
    static_assert(QUANT_GROUP == 8);

    constexpr int64_t QUANT_GROUP_SHIFT = 3;

    // The elements in Q, K, and V will be evenly distributed across each thread group.
    T local_q[VEC_SIZE * VEC_LEN]; // 2 * 8

    const int64_t warp_id       = threadIdx.x / WARP_SIZE;
    const int64_t warp_lane_id  = threadIdx.x % WARP_SIZE;
    const int64_t group_id      = warp_lane_id / THREAD_GROUP_SIZE;
    const int64_t group_lane_id = warp_lane_id % THREAD_GROUP_SIZE;
    const int64_t kv_head_idx     = head_idx / gqa_group_size;

    #pragma unroll
    for (int64_t i = 0; i < VEC_LEN; i++) {
        // copy 128(16 * 8) bits from Q to Local Q

        // 这个地方是错开间隔读取的，不知道如果设置成为连续位置读取会不会一样呢？
        copy<sizeof(T) * VEC_SIZE>(
            &query[
                batch_idx * query_stride_s +
                head_idx * query_stride_h +
                (group_lane_id + i * THREAD_GROUP_SIZE) * VEC_SIZE
            ],
            &local_q[i * VEC_SIZE]);
    }
    // ------------------------------------------------ //
    // Step 2. Solve QK Dot

    const int64_t context_len = seq_len;
    extern __shared__ float logits[];
    float qk_max = -FLT_MAX;

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        int8_t local_k_quant[VEC_SIZE * VEC_LEN];
        T local_k[VEC_SIZE * VEC_LEN];
        T local_k_scale[VEC_LEN];
        const int64_t context_id = base_id + group_id;
        const int64_t mem_context_id = *(b_start_loc + context_id);

        // all thread groups within a warp must be launched together.
        if (context_id >= context_len){
            memset(local_k, 0, sizeof(local_k));
        } else {
            const int64_t key_offset
                            = (mem_context_id) * kcache_stride_s
                            + kv_head_idx * kcache_stride_h
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from K to Local K
                const int64_t key_idx = key_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&k_cache[key_idx],  &local_k_quant[i * VEC_SIZE]);

                const int64_t key_scale_idx = key_idx >> QUANT_GROUP_SHIFT;
                local_k_scale[i] = k_scale[key_scale_idx];
            }

            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_k[i * VEC_SIZE + j]
                        = local_k_scale[i] * (T)local_k_quant[i * VEC_SIZE + j];
                }
            }
        }

        // Ready for QK Dot
        const float qk_dot
            = attn_scale
            * attn_thread_group_dot<THREAD_GROUP_SIZE, VEC_LEN * VEC_SIZE>(local_q, local_k);

        if (group_lane_id == 0 && context_id < context_len) {
            logits[context_id] = qk_dot;
            qk_max = fmaxf(qk_dot, qk_max);
        }
    }

    // ------------------------------------------------ //
    // Step 3. Softmax

    __shared__ float red_smem[WPT];

    qk_max = attn_block_reduce_max<WPT>(qk_max, red_smem);

    float exp_sum = 0.0f;
    for (int64_t context_id = threadIdx.x; context_id < context_len; context_id += TPB){
        logits[context_id] -= qk_max;
        logits[context_id] = exp(logits[context_id]);
        exp_sum += logits[context_id];
    }

    static_assert(WPT == 2 || WPT == 4 || WPT == 8 || WPT == 16 || WPT == 32 || WPT == 64);
    exp_sum = attn_block_reduce_sum<WPT>(exp_sum, red_smem);

    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int64_t context_id = threadIdx.x; context_id < context_len; context_id += TPB) {
        logits[context_id] *= inv_sum;
    }
    __syncthreads(); // Must have this.

    // ------------------------------------------------ //
    // Step 4. Solve logits * V

    int8_t local_v_quant[VEC_SIZE * VEC_LEN];
    float local_v[VEC_SIZE * VEC_LEN];
    T local_v_scale[VEC_LEN];

    #pragma unroll
    for(int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        local_v[i] = 0;
    }

    for (int64_t base_id = warp_id * GPW; base_id < context_len; base_id += GPT) {
        const int64_t context_id = base_id + group_id;
        const int64_t mem_context_id = *(b_start_loc + context_id);
        // all thread groups within a warp must be launched together.
        if (context_id < context_len){
            const int64_t value_offset
                            = (mem_context_id) * vcache_stride_s
                            + kv_head_idx * vcache_stride_h
                            + group_lane_id * VEC_SIZE;
            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                // copy 128(16 * 8) bits from V to Local V
                const int64_t value_idx = value_offset + i * THREAD_GROUP_SIZE * VEC_SIZE;
                copy<sizeof(int8_t) * VEC_SIZE>(&v_cache[value_idx],  &local_v_quant[i * VEC_SIZE]);

                const int64_t value_scale_idx = value_idx >> QUANT_GROUP_SHIFT;
                local_v_scale[i] = v_scale[value_scale_idx];
            }

            #pragma unroll
            for (int64_t i = 0; i < VEC_LEN; i++) {
                #pragma unroll
                for (int64_t j = 0; j < VEC_SIZE; j++) {
                    local_v[i * VEC_SIZE + j] += (tofloat(local_v_scale[i])
                                                * (float)local_v_quant[i * VEC_SIZE + j]
                                                * logits[context_id]);
                }
            }
        }
    }

    #pragma unroll
    for (int32_t i = 0; i < VEC_SIZE * VEC_LEN; i++) {
        #pragma unroll
        for (int32_t mask = THREAD_GROUP_SIZE; mask <= WARP_SIZE >> 1; mask = mask << 1) {
            local_v[i] += __shfl_xor_sync(uint32_t(-1), local_v[i], mask);
        }
    }

    __syncthreads();

    // do some reuse
    for (int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        logits[i] = 0;
    }

    __syncthreads();

    if (warp_lane_id < THREAD_GROUP_SIZE) {
        #pragma unroll
        for (int32_t i = 0; i < VEC_LEN; i++) {
            #pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; j++) {
                atomicAdd(
                    logits + i * THREAD_GROUP_SIZE * VEC_SIZE + warp_lane_id * VEC_SIZE + j,
                    local_v[i * VEC_SIZE + j]
                );
            }
        }
    }

    __syncthreads();

    for (int64_t i = threadIdx.x; i < HEAD_SIZE; i += TPB){
        output[batch_idx * output_stride_s + head_idx * output_stride_h + i] = logits[i];
    }
}


template<typename T>
void run_group_int8kv_decode_attention_kernel(
    T* __restrict__ output,         
    const T* __restrict__ query,    
    const int8_t* k_cache,              
    const T* k_scale,                 
    const int8_t* v_cache,
    const T* v_scale,
    const float attn_scale,
    const int64_t output_stride_s,
    const int64_t output_stride_h,
    const int64_t query_stride_s,
    const int64_t query_stride_h,
    const int64_t kcache_stride_s,
    const int64_t kcache_stride_h,
    const int64_t vcache_stride_s,
    const int64_t vcache_stride_h,
    const int32_t * __restrict__ b_seq_len,
    const int32_t * __restrict__ b_req_idx,
    const int32_t * __restrict__ req_to_tokens,
    const int64_t req_to_tokens_stride,
    const int64_t max_len_in_batch,

    const int64_t batch_size,
    const int64_t q_head_num,
    const int64_t head_dim,
    const int64_t gqa_group_size) {

    constexpr int64_t WARP_SIZE = 32;
    constexpr int64_t TPB = 256;
    constexpr int64_t MAX_SHM_SIZE = 48 * 1024;

    constexpr int64_t reduce_shm_size = TPB / WARP_SIZE * sizeof(float);
    const int64_t logits_size = max(max_len_in_batch * sizeof(float), head_dim * sizeof(float));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (reduce_shm_size + logits_size <= MAX_SHM_SIZE) {
        const dim3 grid_size = {(unsigned int)q_head_num, (unsigned int)batch_size, 1};
        switch (head_dim){
            case 64:
                dynamic_batching_decoding_cache_attention_fp16_kernel<64, 4, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>
                (
                    output, query, k_cache, k_scale, v_cache, v_scale,
                    attn_scale,
                    output_stride_s, output_stride_h,
                    query_stride_s, query_stride_h,
                    kcache_stride_s, kcache_stride_h,
                    vcache_stride_s, vcache_stride_h,
                    b_seq_len, b_req_idx, req_to_tokens,
                    req_to_tokens_stride,
                    max_len_in_batch,
                    gqa_group_size
                );
                break;
            case 96:
                dynamic_batching_decoding_cache_attention_fp16_kernel<96, 4, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>
                (
                    output, query, k_cache, k_scale, v_cache, v_scale,
                    attn_scale,
                    output_stride_s, output_stride_h,
                    query_stride_s, query_stride_h,
                    kcache_stride_s, kcache_stride_h,
                    vcache_stride_s, vcache_stride_h,
                    b_seq_len, b_req_idx, req_to_tokens,
                    req_to_tokens_stride,
                    max_len_in_batch,
                    gqa_group_size
                );
                break;
            case 128:
                dynamic_batching_decoding_cache_attention_fp16_kernel<128, 8, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>
                (
                    output, query, k_cache, k_scale, v_cache, v_scale,
                    attn_scale,
                    output_stride_s, output_stride_h,
                    query_stride_s, query_stride_h,
                    kcache_stride_s, kcache_stride_h,
                    vcache_stride_s, vcache_stride_h,
                    b_seq_len, b_req_idx, req_to_tokens,
                    req_to_tokens_stride,
                    max_len_in_batch,
                    gqa_group_size
                );
                break;
            case 256:
                dynamic_batching_decoding_cache_attention_fp16_kernel<256, 16, 256, 8>
                <<<grid_size, 256, logits_size, stream>>>
                (
                    output, query, k_cache, k_scale, v_cache, v_scale,
                    attn_scale,
                    output_stride_s, output_stride_h,
                    query_stride_s, query_stride_h,
                    kcache_stride_s, kcache_stride_h,
                    vcache_stride_s, vcache_stride_h,
                    b_seq_len, b_req_idx, req_to_tokens,
                    req_to_tokens_stride,
                    max_len_in_batch,
                    gqa_group_size
                );
                break;
            default:
                assert(false);
        }
    } else {
        assert(false);
    }
}

void group_int8kv_decode_attention(at::Tensor o, at::Tensor q, at::Tensor k, at::Tensor k_s,  at::Tensor v,  at::Tensor v_s, at::Tensor req_to_tokens, at::Tensor b_req_idx, at::Tensor b_seq_len, int max_len_in_batch) {
    int64_t batch_size = b_seq_len.sizes()[0];
    int64_t head_num = q.sizes()[1];
    int64_t head_dim = q.sizes()[2]; // q shape [batchsize, head_num, head_dim]
    float att_scale = 1.0 / std::sqrt(head_dim);
    int64_t kv_head_num = k.sizes()[1];
    assert(head_num % kv_head_num == 0);
    int64_t gqa_group_size = head_num / kv_head_num;
    LIGHT_DISPATCH_FLOATING_TYPES(q.scalar_type(), "group_int8kv_decode_attention", ([&]{
            run_group_int8kv_decode_attention_kernel<scalar_t>(
                o.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), 
                k.data_ptr<int8_t>(), k_s.data_ptr<scalar_t>(),
                v.data_ptr<int8_t>(), v_s.data_ptr<scalar_t>(),
                att_scale,
                o.stride(0),
                o.stride(1),
                q.stride(0),
                q.stride(1),
                k.stride(0),
                k.stride(1),
                v.stride(0),
                v.stride(1),
                b_seq_len.data_ptr<int32_t>(),
                b_req_idx.data_ptr<int32_t>(),
                req_to_tokens.data_ptr<int32_t>(),
                req_to_tokens.stride(0),
                max_len_in_batch,
                batch_size,
                head_num,
                head_dim,
                gqa_group_size
            );
        }
    ));
}

void group_int8kv_decode_attention(
    torch::Tensor o, 
    torch::Tensor q, 
    torch::Tensor k, 
    torch::Tensor k_s,  
    torch::Tensor v,  
    torch::Tensor v_s, 
    torch::Tensor req_to_tokens, 
    torch::Tensor b_req_idx, 
    torch::Tensor b_seq_len, 
    int64_t max_len_in_batch)
{
    group_int8kv_decode_attention(
        o,
        q, 
        k, 
        k_s, 
        v, 
        v_s, 
        req_to_tokens, 
        b_req_idx, 
        b_seq_len, 
        static_cast<int>(max_len_in_batch)
    );
}


}
}