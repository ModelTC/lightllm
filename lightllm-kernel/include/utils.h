
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

// mycuda, some wrappers and utils
namespace lightllm {
// type definitions
using fp16_t = __half;
using fp16x2_t = __half2;
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;

using fp8_e4m3_t = __nv_fp8_e4m3;
using fp8x2_e4m3_t = __nv_fp8x2_e4m3;
using fp8x4_e4m3_t = __nv_fp8x4_e4m3;

using fp32_t = float;
using fp32x2_t = float2;
using fp32x4_t = float4;

using int32x4_t = int4;
using int32x2_t = int2;

using int8x2_t = short;
using int8x4_t = int32_t;
using int8x8_t = int64_t;

using vec_type = int4;

// convert fp16_t to fp32_t
__device__ inline fp32_t cvt_f16_f32(const fp16_t x) { return __half2float(x); }

__device__ inline fp16_t cvt_f32_f16(const fp32_t x) { return __float2half(x); }

// Convert bf16_t to fp32_t
__device__ inline fp32_t cvt_bf16_f32(const bf16_t x) {
    return __bfloat162float(x);
}

// Convert fp32_t to bf16_t
__device__ inline bf16_t cvt_f32_bf16(const fp32_t x) {
    return __float2bfloat16(x);
}

// bf16x2 to fp32x2 conversion
__device__ inline fp32x2_t bf16x2_to_fp32x2(bf16x2_t bf16x2_val) {
    // Extract the two bfloat16 values from bf16x2
    bf16_t low = __low2bfloat16(bf16x2_val);
    bf16_t high = __high2bfloat16(bf16x2_val);

    // Convert bfloat16 to float
    float low_f = __bfloat162float(low);
    float high_f = __bfloat162float(high);

    // Pack the two floats into a float2
    return make_float2(low_f, high_f);
}

__device__ inline bf16x2_t _float22bf162_rn(fp32x2_t val) {
    bf16_t low = __float2bfloat16(val.x);
    bf16_t high = __float2bfloat16(val.y);
    return bf16x2_t(low, high);
}

__device__ inline int8_t float_to_int8_rn(fp32_t x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
}

template <typename T>
__host__ __device__ T Cdiv(T numerator, T denominator) {
    return (numerator + denominator - 1) / denominator;
}

template <typename T>
__host__ __device__ T Adiv(T value, T alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

__device__ inline fp32x2_t operator+(const fp32x2_t& a, const fp32x2_t& b) {
    return {a.x + b.x, a.y + b.y};
}

__device__ inline fp16_t abs(const fp16_t& x) { return __habs(x); }

__device__ inline bool operator>(const fp16_t& a, const fp16_t& b) {
    return __hgt(a, b);
}

__device__ inline fp16_t operator+(const fp16_t& a, const fp16_t& b) {
    return __hadd(a, b);
}

__device__ inline fp16_t operator-(const fp16_t& a, const fp16_t& b) {
    return __hsub(a, b);
}

__device__ inline fp16_t operator*(const fp16_t& a, const fp16_t& b) {
    return __hmul(a, b);
}

__device__ inline fp16_t operator/(const fp16_t& a, const fp16_t& b) {
    return __hdiv(a, b);
}

__device__ inline fp16_t& operator+=(fp16_t& a, const fp16_t& b) {
    a = __hadd(a, b);
    return a;
}

__device__ inline fp16_t& operator-=(fp16_t& a, const fp16_t& b) {
    a = __hsub(a, b);
    return a;
}

__device__ inline fp16_t& operator*=(fp16_t& a, const fp16_t& b) {
    a = __hmul(a, b);
    return a;
}

__device__ inline fp16_t& operator/=(fp16_t& a, const fp16_t& b) {
    a = __hdiv(a, b);
    return a;
}

__device__ inline fp16x2_t operator+(const fp16x2_t& a, const fp16x2_t& b) {
    return __hadd2(a, b);
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
__device__ inline void vec_copy(const void* src, void* dest)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(src);
    T* out = static_cast<T*>(dest);
    *out = *in;
}

template<int32_t divisor>
__device__ inline int32x2_t divmod(const int32_t x);

template<>
__device__ inline int32x2_t divmod<128>(const int32_t x) {
    return {x >> 7, x & 0x7F};
}

template<>
__device__ inline int32x2_t divmod<64>(const int32_t x) {
    return {x >> 6, x & 0x3F};
}

template<>
__device__ inline int32x2_t divmod<32>(const int32_t x) {
    return {x >> 5, x & 0x1F};
}

template<>
__device__ inline int32x2_t divmod<16>(const int32_t x) {
    return {x >> 4, x & 0x0F};
}

template<>
__device__ inline int32x2_t divmod<8>(const int32_t x) {
    return {x >> 3, x & 0x07};
}

template<>
__device__ inline int32x2_t divmod<4>(const int32_t x) {
    return {x >> 2, x & 0x03};
}

template<>
__device__ inline int32x2_t divmod<2>(const int32_t x) {
    return {x >> 1, x & 0x01};
}

}  // namespace lightllm

// mytorch, some wrappers and utils
namespace lightllm {
using Tensor = torch::Tensor;

template <typename T>
__host__ inline T *PTR(at::Tensor t) {
    return reinterpret_cast<T *>(t.data_ptr());
}

template <>
__host__ inline fp16_t *PTR(at::Tensor t) {
    return reinterpret_cast<fp16_t *>(t.data_ptr());
}

template <>
__host__ inline fp16x2_t *PTR(at::Tensor t) {
    return reinterpret_cast<fp16x2_t *>(t.data_ptr());
}

template <>
__host__ inline int8x4_t *PTR(at::Tensor t) {
    return reinterpret_cast<int8x4_t *>(t.data_ptr());
}

template <>
__host__ inline int8x2_t *PTR(at::Tensor t) {
    return reinterpret_cast<int8x2_t *>(t.data_ptr());
}

template <>
__host__ inline int8_t *PTR(at::Tensor t) {
    return reinterpret_cast<int8_t *>(t.data_ptr());
}

template <>
__host__ inline uint16_t *PTR(at::Tensor t) {
    return reinterpret_cast<uint16_t *>(t.data_ptr());
}

template <>
__host__ inline uint32_t *PTR(at::Tensor t) {
    return reinterpret_cast<uint32_t *>(t.data_ptr());
}

template <>
__host__ inline void *PTR(at::Tensor t) {
    return reinterpret_cast<void *>(t.data_ptr());
}

__device__ inline
void block_debug_print_matrix(fp16_t *ptr, int32_t M, int32_t N, int32_t stride) {
    if(threadIdx.x == 0) {
        printf("Debug Matrix [%d, %d, %d]: \n", blockIdx.x, blockIdx.y, blockIdx.z);
        for(int32_t i = 0; i < M; i++) {
            for(int32_t j = 0; j < N; j++) {
                printf("%.2f ", __half2float(ptr[i * stride + j]));
            }
            printf("\n");
        }
    }
}

}  // namespace lightllm
