#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <type_traits>
#include "common.h"
#include "verify.h"

// Casts an fp16 input to the restricted values of float4_e2m1,
// that is to say [0., 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0].
template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias>
float_type fp16_to_fp4_simulate(float_type* val) {
    uint16_t val_view = *(uint16_t*)val;

    uint16_t exp = val_view >> half_mantissa_bits;
    exp = exp & ((1 << half_exp_bits) - 1);

    bool sign = (val_view >> (half_mantissa_bits + half_exp_bits)) & 1;

    bool mantissa_last = (val_view >> (half_mantissa_bits - 1)) & 1;

    int16_t exp_unbias = exp - half_exp_bias;
    int16_t new_exp = exp_unbias + FLOAT4_EXP_BIAS;

    int16_t exp_shift = (new_exp <= 0) * (1 - new_exp);

    // Typically 9.
    // Take the min to prevent overflow on `uint16_t half`. This is the case for very small values,
    // correctly mapped to `round_close`.
    uint16_t tail_bits =
        sycl::min(16U, half_mantissa_bits - FLOAT4_MANTISSA_BITS + exp_shift);

    uint16_t mantissa_plus_one = val_view & ((1 << (half_mantissa_bits + 1)) - 1);

    uint16_t half = 1 << (tail_bits - 1);

    uint16_t tail = mantissa_plus_one & ((1 << tail_bits) - 1);

    bool round_close = (tail < half);  // round towards 0
    bool round_away = (tail > half);  // round away from 0
    bool tie = tail == half;

    uint16_t new_mantissa;

    bool new_mantissa_close = 0;
    uint16_t new_exp_close = 0;

    bool new_mantissa_away = 0;
    uint16_t new_exp_away = 0;

    uint16_t new_exp_tie = 0;

    // # 1. round down
    // if new_exp == 0: # case [0.5, 0.749999]
    //     new_mantissa = 0
    // elif new_exp < 0:  # case [0, 0.24999]
    //     new_mantissa = 0
    // else:
    //     new_mantissa = mantissa_last

    new_mantissa_close = (new_exp > 0) * mantissa_last;
    new_exp_close = exp;

    // # 2. round up
    // if new_exp <= 0:  # case [0.250001, 0.499999] and [0.75001, 0.99999]
    //     new_mantissa = 0
    //     new_exp += 1
    // elif mantissa_last == 0:
    //     new_mantissa = 1
    // else:
    //     new_mantissa = 0
    //     new_exp += 1

    new_mantissa_away = (new_exp > 0) && (mantissa_last == 0);
    new_exp_away = exp + ((new_exp <= 0) || (mantissa_last == 1));

    // # 3. tie
    // 0.25 -> 0. (handled by `exp > (half_exp_bias - 2)`)
    // 0.75 -> 1.
    // 1.25 -> 1.
    // 1.75 -> 2.
    // 2.5 -> 2.
    // 3.5 -> 4.
    // 5. -> 4.
    new_exp_tie = (exp > (half_exp_bias - 2)) * (exp + (mantissa_last == 1));

    // # Gather round up, round down and tie.
    new_exp = round_away * new_exp_away + round_close * new_exp_close + tie * new_exp_tie;
    new_mantissa = round_away * new_mantissa_away + round_close * new_mantissa_close;

    // if new_exp > 3:
    //     new_mantissa = 1
    new_mantissa = new_mantissa + (new_exp > (2 + half_exp_bias)) * (new_mantissa == 0);

    // Clamp the exponent to acceptable values.
    new_exp =
        (new_exp >= (half_exp_bias - 2)) *
        sycl::max((half_exp_bias - 2), sycl::min((uint32_t)new_exp, half_exp_bias + 2));

    uint16_t qdq_val = (sign << 15) + (new_exp << half_mantissa_bits) + (new_mantissa << (half_mantissa_bits - 1));
    float_type result = *(float_type*)(&qdq_val);
    return result;
}

template<typename float_type, uint32_t half_exp_bits, uint32_t half_mantissa_bits, uint32_t half_exp_bias, uint16_t val_to_add, uint16_t sign_exponent_mask>
void qdq_mxfp4_kernel(const float_type* inp, float_type* out,
                      const sycl::nd_item<3> &item) {
    if constexpr (std::is_same_v<float_type, sycl::ext::oneapi::bfloat16>) {
       using namespace sycl::ext::oneapi::experimental;
    } else {
       using namespace sycl;
    }
    // Each thread handles one element.

    int idx = item.get_global_id(2);
    float_type elem = inp[idx];
    float_type block_max = fabs(elem);

    // Compute the max 32 lanes by 32 lanes.
    // Each thread handles a single value, thus applying `shfl_xor` 5 times.
    // (Max over 2**5 = 32 values).
    for (int i = 1; i < 32; i*=2) {
        block_max = fmax(block_max, shfl_xor_bf16_or_half(block_max, i, item));
    }

    // Apply rounding strategy to block_max.
    // cannot take the address of an rvalue so need this intermediate `block_max_uint` variable?
    uint16_t block_max_uint = (*(uint16_t*)(&block_max) + val_to_add) & sign_exponent_mask;

    block_max = *(float_type*)(&block_max_uint);

    uint8_t scale_exp =
        sycl::max(0, FLOAT8_E8M0_MAX_EXP +
                  sycl::min(bf16_or_half2int_rn<float_type>(floor(log2(block_max))) - 2, FLOAT8_E8M0_MAX_EXP));
    float_type scale = float_to_bf16_or_half<float_type>(sycl::pow(2.f, scale_exp - FLOAT8_E8M0_MAX_EXP));

    elem = elem / scale;

    float_type elem_fp4 = fp16_to_fp4_simulate<float_type, half_exp_bits, half_mantissa_bits, half_exp_bias>(&elem);

    out[idx] = elem_fp4 * scale;
}

template <typename T>
void mxfp4(sycl::queue &q, T *y, const T *a, int numel, int niters, int group_size = 32) {

  int block_size;

  if (numel % 128 == 0) {
      block_size = 128;
  } else if (numel % 64 == 0) {
      block_size = 64;
  } else {
      printf("Expected qdq_mxfp4 input number of elements to be a multiple of 64, but it is not!");
      return;
  }

  sycl::range<3> gws (1, 1, numel);
  sycl::range<3> lws (1, 1, block_size);

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    if constexpr (std::is_same_v<T, sycl::half>) {
      q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            qdq_mxfp4_kernel<sycl::half, FLOAT16_EXP_BITS,
                             FLOAT16_MANTISSA_BITS, FLOAT16_EXP_BIAS,
                             FLOAT16_VAL_TO_ADD, FLOAT16_SIGN_EXPONENT_MASK>(
                a, y, item);
          });
    } else if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
      q.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            qdq_mxfp4_kernel<sycl::ext::oneapi::bfloat16, BFLOAT16_EXP_BITS,
                             BFLOAT16_MANTISSA_BITS, BFLOAT16_EXP_BIAS,
                             BFLOAT16_VAL_TO_ADD, BFLOAT16_SIGN_EXPONENT_MASK>(
                a, y, item);
          });
    }
    else {
        printf("Wrong input dtype in qdq_mxfp4!");
    }
  }
  q.wait();
  auto end = std::chrono::high_resolution_clock::now();
  double time = std::chrono::duration_cast<std::chrono::microseconds>
                (end - start).count() / niters / 1.0e6;
  double size = 2.0 * sizeof(T) * numel / 1.0e9;
  printf("size(GB):%.2f, average time(sec):%f, Bandwidth (GB/sec):%f\n",
         size, time, size / time);
}


template <typename T>
void mxfp4_sim(sycl::queue &q, int numel, int niters)
{
  size_t bytes = (size_t)numel * sizeof(T); 
  T *src, *dst;
  src = (T*) malloc (bytes);
  dst = (T*) malloc (bytes);

  // basic functional test 
  const float v[] = {0.0, 0.5, 1.0, 1.5, 2, 3, 4,
                     -0.0, -0.5, -1.0, -1.5, -2, -3, -4,
                     0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5,
                     0.1, 0.3, 0.4, 0.6, 0.9,
                     -0.1, -0.3, -0.4, -0.6, -0.9};

  srand(123);
  for (int i = 0; i < numel / 32; i++) {
    for (int j = i*32; j < (i+1)*32; j++)
      src[j] = v[rand() % (sizeof(v) / sizeof(v[0]))];
    int r = rand() % 32;
    src[i*32 + r] = r < 16 ? 6 : -6;
  }

  T *d_src, *d_dst;
  d_src = (T *)sycl::malloc_device(bytes, q);
  d_dst = (T *)sycl::malloc_device(bytes, q);

  q.memcpy(d_src, src, bytes).wait();

  mxfp4<T>(q, d_dst, d_src, numel, niters);

  q.memcpy(dst, d_dst, bytes).wait();

  verify(src, dst, numel);

  free(src);
  free(dst);
  sycl::free(d_src, q);
  sycl::free(d_dst, q);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int numel = atoi(argv[1]);
  const int niters = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  mxfp4_sim<sycl::half>(q, numel, niters);
  mxfp4_sim<sycl::ext::oneapi::bfloat16>(q, numel, niters);
  return 0;
}
