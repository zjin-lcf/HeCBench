// Check for bfloat16 support
// V100 is compute capability 7.0 and doesn't support __nv_bfloat16 (requires
// >= 8.0) ROCm/HIP generally supports bfloat16 on modern GPUs

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>

#define FLOAT16_MANTISSA_BITS 10
#define FLOAT16_EXP_BITS 5
#define FLOAT16_EXP_BIAS 15

#define FLOAT4_MANTISSA_BITS 1
#define FLOAT4_EXP_BITS 2
#define FLOAT4_EXP_BIAS 1

#define FLOAT8_E8M0_MAX_EXP 127

#define BFLOAT16_MANTISSA_BITS 7
#define BFLOAT16_EXP_BITS 8
#define BFLOAT16_EXP_BIAS 127

#define FLOAT16_VAL_TO_ADD \
  (1 << (FLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define FLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (FLOAT16_EXP_BITS + 1)) - 1) << FLOAT16_MANTISSA_BITS)

#define BFLOAT16_VAL_TO_ADD \
  (1 << (BFLOAT16_MANTISSA_BITS - FLOAT4_MANTISSA_BITS - 1))
#define BFLOAT16_SIGN_EXPONENT_MASK \
  (((1 << (BFLOAT16_EXP_BITS + 1)) - 1) << BFLOAT16_MANTISSA_BITS)

template <typename T>
__device__ int bf16_or_half2int_rn(const T h);

template <typename T>
__device__ T float_to_bf16_or_half(const float x);

template <typename T>
__device__ float bf16_or_half_to_float(const T x);

template <typename T>
inline __device__ T shfl_xor_bf16_or_half(T x, int laneMask) {
  return __shfl_xor_sync(0xffffffff, x, laneMask);
}

// Definitions

template <>
inline __device__ int bf16_or_half2int_rn(const __half h) {
  return __half2int_rn(h);
}

template <>
inline __device__ int bf16_or_half2int_rn(const __nv_bfloat16 h) {
  return __bfloat162int_rn(h);
}

template <>
inline __device__ __half float_to_bf16_or_half(const float x) {
  return __float2half(x);
}

template <>
inline __device__ __nv_bfloat16 float_to_bf16_or_half(const float x) {
  return __float2bfloat16(x);
}

template <>
inline __device__ float bf16_or_half_to_float(const __half x) {
  return __half2float(x);
}

template <>
inline __device__ float bf16_or_half_to_float(const __nv_bfloat16 x) {
  return __bfloat162float(x);
}
