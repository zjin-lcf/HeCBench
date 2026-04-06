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
inline __device__ uint8_t fp8e5m2fnuz_from_fp32_value(float f) {
  /*
   * Binary representation of 65536.0f, which is the first value not
   * representable (i.e. the first value which would overflow in to the sign
   * bit, resulting in a NaN) in fp8e4m3fnuz range:
   * 1 00000 00 - fp8e5m2fnuz
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x8F) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2fnuz normal range
   * into denormalized representation.
   * magic number: ((127 - 16) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

  uint32_t f_bits = fp32_to_bits(f);
  uint32_t result = 0u;

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

  if (f_bits >= fnuz_max) {
    return 0x80; // NaN -- sign bit set to 1, rest 0s
  }

  if (f_bits < (UINT32_C(0x70) << 23) /* 2^-15 in float32 */) {
    // Input exponent is less than -15, the smallest e5m2fnuz exponent, so the
    // number will become subnormal.
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (f_bits >> 21) & 1;

    // update exponent, rounding bias part 1
    f_bits += ((uint32_t)(16 - 127) << 23) + 0xFFFFF;

    // rounding bias part 2
    f_bits += mant_odd;

    // take the bits!
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  result |= sign >> 24;
  return result;
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FNUZ format, in bit representation.
 */
inline __device__ uint8_t fp8e4m3fnuz_from_fp32_value(float f) {
  /*
   * Binary representation of 256.0f, which is the first value not representable
   * (i.e. the first value which would overflow in to the sign bit, resulting in
   * a NaN) in fp8e4m3fnuz range:
   * 1 0000 000 - fp8e4m3fnuz
   * 0 10000111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
   * into denorm representation
   * magic number: ((127 - 8) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint32_t result = 0u;

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

  if (f_bits >= fnuz_max) {
    return 0x80; // NaN -- sign bit set to 1, rest 0s.
  }

  if (f_bits < (UINT32_C(0x78) << 23) /* 2^-7 in float32 */) {
    // Input exponent is less than -7, the smallest e4m3fnuz exponent, so the
    // number will become subnormal.
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (f_bits >> 20) & 1;

    // update exponent, rounding bias part 1
    f_bits += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

    // rounding bias part 2
    f_bits += mant_odd;

    // take the bits!
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= sign >> 24;
  return result;
}

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e5m2fnuz (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e5m2fnuz_from_fp32_value(src[i]);
  }
}

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e4m3fnuz (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = fp8e4m3fnuz_from_fp32_value(src[i]);
  }
}


template <typename Td, typename Ts>
__global__
void fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __hip_cvt_float_to_fp8(src[i], __HIP_SATFINITE, __HIP_E4M3_FNUZ); // CDNA3
  }
}

template <typename Td, typename Ts>
__global__
void fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = __hip_cvt_float_to_fp8(src[i], __HIP_SATFINITE, __HIP_E5M2_FNUZ); // CDNA3 
  }
}
