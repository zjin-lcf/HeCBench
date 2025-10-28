#include <math.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <sycl/sycl.hpp>

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
int bf16_or_half2int_rn(const T h) {
  return sycl::vec<T, 1>(h)
      .template convert<int, sycl::rounding_mode::rte>()[0];
}

template <typename T>
T float_to_bf16_or_half(const float x) {
  return sycl::vec<float, 1>(x)
      .template convert<T, sycl::rounding_mode::automatic>()[0];
}

template <typename T>
float bf16_or_half_to_float(const T x) {
  return sycl::vec<T, 1>(x)
      .template convert<float, sycl::rounding_mode::automatic>()[0];
}

template <typename T>
T shfl_xor_bf16_or_half(T x, int laneMask, const sycl::nd_item<3> &item) {
  return sycl::permute_group_by_xor(item.get_sub_group(), x, laneMask);
}
