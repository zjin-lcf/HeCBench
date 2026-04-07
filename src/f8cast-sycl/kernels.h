#include <cstdint>

inline float fp32_from_bits(uint32_t w) {
  return sycl::bit_cast<float>(w);
}

inline uint32_t fp32_to_bits(float f) {
  return sycl::bit_cast<uint32_t>(f);
}

// Reference https://www.xyzzhangfan.tech/blog/2025/
uint8_t FP32_FP8_E5M2_ref(float value) {
    uint32_t x = fp32_to_bits(value);
    uint32_t sign = (x >> 31) << 7;
    uint32_t bexp = ((x >> 23) & 0xFF);
    uint32_t mantissa = x & 0x7FFFFF;

    if (bexp == 0xFF) {
        // NaN if mant != 0, Inf otherwise
        return (mantissa != 0) ? 0x7F : (sign ? 0xFC : 0x7C);
    }
    int32_t exponent = bexp - 127;

    if (exponent != -127) {
        mantissa |= 0x800000; // Add the leading 1 bit for normalized values
    }

    // Apply the bias for e5m2 (bias of 15)
    int32_t e5m2_exponent = exponent + 15; // Bias for e5m2 is 15

    // Handle exponent out of range(e5m2 has 5 exponent bis, max is 31)
    if (e5m2_exponent > 31) {
        return sign | 0x7C;
    } else if ((e5m2_exponent >= -1) && (e5m2_exponent <= 0)) {
        uint8_t shift_bits = (2 + e5m2_exponent);
        uint8_t e5m2_mantissa = (mantissa >> (24 - shift_bits)) & (0x3 >> (0 - e5m2_exponent));
        return sign | e5m2_mantissa;
    } else if (e5m2_exponent < -1) {
        return sign;
    }

    uint8_t e5m2_mantissa = (mantissa >> 21) & 0x3; //  the leading 1 bit

    uint8_t result = sign | (e5m2_exponent << 2) | e5m2_mantissa;
    return result;
}



uint8_t FP32_FP8_E4M3_ref(float value) {
    uint32_t x = fp32_to_bits(value);
    uint32_t sign = (x >> 31) << 7;
    uint32_t bexp = (x >> 23) & 0xFF;
    uint32_t mantissa = x & 0x7FFFFF;

    if (bexp == 0xFF) {
        // NaN if mant != 0, Inf otherwise
        return (mantissa != 0) ? 0x7F : (sign ? 0xFF : 0x7F);
    }
    int32_t exponent = bexp - 127;

    if (exponent != -127) {
        mantissa |= 0x800000; // Add the leading 1 bit for normalized values
    }

     // Apply the bias for e4m3 (bias of 7)
     int32_t e4m3_exponent = exponent + 7;  // Bias for e4m3 is 7

     // Handle exponent out of range (e4m3 has 4 exponent bits, max is 15)
     if (e4m3_exponent > 15) {
         return sign | 0x7F;  // Saturate to the NaN
         //return sign << 7 | 0x7E;  // Saturate to the max value (S.1111.110 = S.448)
     } else if ((e4m3_exponent > -3) && (e4m3_exponent <= 0)) {
         // Subnormal numbers (exponent is 0)
         // For subnormal numbers, the mantissa is shifted and stored directly
         uint8_t shift_bits = (3 + e4m3_exponent);
         // Add 1 bit in front of mantissa then shift it
         uint8_t e4m3_mantissa = (mantissa >> (24 - shift_bits)) & (0x7 >> (0-e4m3_exponent));  //Shift to get the top N bits (3 to 1)
         return sign | e4m3_mantissa;  // Exponent is 0 for subnormal numbers
     } else if (e4m3_exponent <= -3) {
         // Saturate to 0
         return sign;
     }

     // For normal numbers, normalize mantissa to fit into 3 bits (e4m3 has 3 bits for mantissa)
     uint8_t e4m3_mantissa = (mantissa >> 20) & 0x07;  // Shift to get the top 3 bits

     // Pack the sign, exponent, and mantissa into an 8-bit value
     uint8_t result = sign | (e4m3_exponent << 3) | e4m3_mantissa;
     return result;
 }

// Reference
// torch/headeronly/util
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline uint8_t fp8e5m2_from_fp32_value(float f) {
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
inline uint8_t fp8e4m3fn_from_fp32_value(float f) {
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

void ref_fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems, sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < nelems) {
    dst[i] = FP32_FP8_E5M2_ref(src[i]);
  }
}

template <typename Td, typename Ts>

void ref_fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems, sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < nelems) {
    dst[i] = FP32_FP8_E4M3_ref(src[i]);
  }
}

template <typename Td, typename Ts>

void fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems, sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < nelems) {
    dst[i] = fp8e4m3fn_from_fp32_value(src[i]);
  }
}

template <typename Td, typename Ts>

void fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems, sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < nelems) {
    dst[i] = fp8e5m2_from_fp32_value(src[i]);
  }
}
