#include <cstdint>

// Reference
// torch/headeronly/util

__device__ inline float fp32_from_bits(uint32_t w) {
  return __uint_as_float((unsigned int)w);
}

__device__ inline uint32_t fp32_to_bits(float f) {
  return (uint32_t)__float_as_uint(f);
}


/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline __device__ uint8_t fp8e5m2_from_fp32_value(float f) {
  /*
   * Binary representation of fp32 infinity
   * 0 11111111 00000000000000000000000
   */
  constexpr uint32_t fp32_inf = UINT32_C(255) << 23;

  /*
   * Binary representation of 65536.0f, which is the first value
   * not representable in fp8e5m2 range:
   * 0 11111 00 - fp8e5m2
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(143) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2 normal range
   * into denorm representation
   * magic number: ((127 - 15) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(134) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = f_bits > fp32_inf ? UINT8_C(0x7F) : UINT8_C(0x7C);
  } else {
    if (f_bits < (UINT32_C(113) << 23)) {
      // Input number is smaller than 2^(-14), which is the smallest
      // fp8e5m2 normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint32_t mant_odd = (f_bits >> 21) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(15 - 127) << 23) + 0xFFFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 21);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}


/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FN format, in bit representation.
 */
inline __device__ uint8_t fp8e4m3fn_from_fp32_value(float f) {
  /*
   * Binary representation of 480.0f, which is the first value
   * not representable in fp8e4m3fn range:
   * 0 1111 111 - fp8e4m3fn
   * 0 10000111 11100000000000000000000 - fp32
   */
  constexpr uint32_t fp8_max = UINT32_C(1087) << 20;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fn normal range
   * into denorm representation
   * magic number: ((127 - 7) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(141) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint8_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN - all exponent and mantissa bits set to 1
    result = 0x7f;
  } else {
    if (f_bits < (UINT32_C(121) << 23)) {
      // Input number is smaller than 2^(-6), which is the smallest
      // fp8e4m3fn normal number
      f_bits =
          fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
      result = static_cast<uint8_t>(f_bits - denorm_mask);
    } else {
      // resulting mantissa is odd
      uint8_t mant_odd = (f_bits >> 20) & 1;

      // update exponent, rounding bias part 1
      f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;

      // rounding bias part 2
      f_bits += mant_odd;

      // take the bits!
      result = static_cast<uint8_t>(f_bits >> 20);
    }
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e5m2_from_fp32_value(src[i]);
  }
}

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e4m3fn_from_fp32_value(src[i]);
  }
}

template <typename Td, typename Ts>
__global__
void fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E4M3);
  }
}

template <typename Td, typename Ts>
__global__
void fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __nv_cvt_float_to_fp8(src[i], __NV_SATFINITE, __NV_E5M2);
  }
}
