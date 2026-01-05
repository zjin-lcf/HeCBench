// Reference https://www.xyzzhangfan.tech/blog/2025/
__device__
uint8_t FP32_FP8_E5M2(float value) {
    if (isnan(value)) return 0xFF; // NaN is 0xFF, 0xFE, 0xFD
    if (isinf(value)) return (value > 0) ? 0x7C : 0xFC; // inf is 0xFC and 0x7C
    // Handle zero case
    if (value == 0) return 0x00;
    // Extract the sign, exponent, and mantissa of the FP32
    uint32_t bits = *reinterpret_cast<uint32_t*>(&value);

    // Extract the sign (1 bit)
    uint8_t sign = (bits >> 31) & 0x1;

    // Extract the exponent (8 bits for FP32)
    int32_t exponent = ((bits >> 23) & 0xFF) - 127; // Bias for FP32 is 127

    // Extract the mantissa (23 bits for FP32)
    uint32_t mantissa = bits & 0x7FFFFF; // Get the frational part

    if (exponent != -127) {
        mantissa |= 0x800000; // Add the leading 1 bit for normalized values
    }

    // Apply the bias for e5m2 (bias of 15)
    int32_t e5m2_exponent = exponent + 15; // Bias for e5m2 is 15

    // Handle exponent out of range(e5m2 has 5 exponent bis, max is 31)
    if (e5m2_exponent > 31) {
        return sign << 7 | 0x7C; // Saturate to inf
    } else if ((e5m2_exponent >= -1) && (e5m2_exponent <= 0)) {
        uint8_t shift_bits = (2 + e5m2_exponent);
        uint8_t e5m2_mantissa = (mantissa >> (24 - shift_bits)) & (0x3 >> (0 - e5m2_exponent));
        return sign << 7 | 0x00 | e5m2_mantissa;
    } else if (e5m2_exponent < -1) {
        return sign << 7 | 0x00;
    }

    uint8_t e5m2_mantissa = (mantissa >> (23 - 2)) & 0x03;

    uint8_t result = (sign << 7) | (e5m2_exponent << 2) | e5m2_mantissa;
    return result;
}


// Reference https://www.xyzzhangfan.tech/blog/2025/
__device__
uint8_t FP32_FP8_E4M3(float value) {
     // Handle special cases (NaN and Infinity)
     if (isnan(value)) return 0xFF;  // NaN is 0xFF in e4m3
     if (isinf(value)) return (value > 0) ? 0x7F : 0xFF;  // Infinity, 0x7F for +Inf, 0xFF for -Inf

     // Handle zero case
     if (value == 0) return 0x00;  // Zero is 0x00 in e4m3
     // Extract the sign, exponent, and mantissa of the FP32
     uint32_t bits = *reinterpret_cast<uint32_t*>(&value);

     // Extract the sign (1 bit)
     uint8_t sign = (bits >> 31) & 0x1;

     // Extract the exponent (8 bits for FP32)
     int32_t exponent = ((bits >> 23) & 0xFF) - 127;  // Bias for FP32 is 127

     // Extract the mantissa (23 bits for FP32)
     uint32_t mantissa = bits & 0x7FFFFF;  // Get the fractional part

     // For normal numbers, add the implicit leading 1 in the mantissa
     if (exponent != -127) {
         mantissa |= 0x800000;  // Add the leading 1 bit for normalized values
     }

     // Apply the bias for e4m3 (bias of 7)
     int32_t e4m3_exponent = exponent + 7;  // Bias for e4m3 is 7

     // Handle exponent out of range (e4m3 has 4 exponent bits, max is 15)
     if (e4m3_exponent > 15) {
         return sign << 7 | 0x7F;  // Saturate to the NaN
         //return sign << 7 | 0x7E;  // Saturate to the max value (S.1111.110 = S.448)
     } else if ((e4m3_exponent > -3) && (e4m3_exponent <= 0)) {
         // Subnormal numbers (exponent is 0)
         // For subnormal numbers, the mantissa is shifted and stored directly
         uint8_t shift_bits = (3 + e4m3_exponent);
         // Add 1 bit in front of mantissa then shift it
         uint8_t e4m3_mantissa = (mantissa >> (24 - shift_bits)) & (0x7 >> (0-e4m3_exponent));  //Shift to get the top N bits (3 to 1)
         return sign << 7 | 0x00 | e4m3_mantissa;  // Exponent is 0 for subnormal numbers
     } else if (e4m3_exponent <= -3) {
         // Saturate to 0
         return sign << 7 | 0x00;
     }

     // For normal numbers, normalize mantissa to fit into 3 bits (e4m3 has 3 bits for mantissa)
     uint8_t e4m3_mantissa = (mantissa >> (23 - 3)) & 0x07;  // Shift to get the top 3 bits

     // Pack the sign, exponent, and mantissa into an 8-bit value
     uint8_t result = (sign << 7) | (e4m3_exponent << 3) | e4m3_mantissa;
     return result;
 }

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e5m2 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = FP32_FP8_E5M2(src[i]);
  }
}

template <typename Td, typename Ts>
__global__
void ref_fp32_cvt_e4m3 (
  Td *__restrict__ dst, const Ts *__restrict__ src, const int nelems)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nelems) {
    dst[i] = FP32_FP8_E4M3(src[i]);
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
