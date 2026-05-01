#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "reference.h"

#pragma omp declare target
uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}
#pragma omp end declare target

template <typename T>
void uniform_fill_kernel(
    T* data,
    uint64_t n,
    float low,
    float high,
    uint32_t seed
) {
  #pragma omp target teams distribute parallel for
  for (uint64_t idx = 0; idx < n; idx++) {

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
  }
}

// Helper functions to convert between different data types
// (float, half, bfloat16) for the merge attention states kernel.
template <typename T>
inline float to_float(T u) {
  return static_cast<float>(u);
}
template <typename T>
inline void from_float(T& d, float s) {
  d = T(s);
}

// Reference
// https://github.com/sgl-project/sgl-kernel/csrc/attention/merge_attn_states.cu
//
// Merges the attention states from prefix and suffix
// into the output tensor. NUM_TOKENS: n, NUM_HEADS: h, HEAD_SIZE: d
//
// @param output [n,h,d] The output tensor to store the merged attention states.
// @param prefix_output [n,h,d] The prefix attention states.
// @param suffix_output [n,h,d] The suffix attention states.
// @param lse [n,h] Optional tensor to store the log-sum-exp values.
// @param prefix_lse [n,h] The log-sum-exp values for the prefix attention states.
// @param suffix_lse [n,h] The log-sum-exp values for the suffix attention states.
//
//
// LSE(I) = log Sum(exp(q * ki)
// O(I) = Sum(exp(q * ki) / exp(LSE(I)) * vi)
//
// output
// exp(LSE(I))O(I) + exp(LSE(J))O(J)
// -----------------------------------
//    exp(LSE(I)) + exp(LSE(J))
//
// lse
// log(exp(LSE(I)) + exp(LSE(J)))
template <typename scalar_t>
void merge_attn_states_kernel(
          scalar_t* __restrict__ output,
    const scalar_t* __restrict__ prefix_output,
    const scalar_t* __restrict__ suffix_output,

          float* __restrict__ lse,
    const float* __restrict__ prefix_lse,
    const float* __restrict__ suffix_lse,

    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_size)
{
    #pragma omp target teams distribute num_teams(num_tokens * num_heads)
    for (int t = 0; t < num_tokens; ++t) {
      for (int h = 0; h < num_heads; ++h) {
            const int lse_idx = t * num_heads + h;

            // exp(-inf)=0, exp(+inf)=NaN
            float p_lse = prefix_lse[lse_idx];
            float s_lse = suffix_lse[lse_idx];

            const float inf = std::numeric_limits<float>::infinity();
            if (p_lse == inf) p_lse = -inf;
            if (s_lse == inf) s_lse = -inf;

            // safe-softmax: subtract the element-wise maximum so that
            // numerical range under control.
            const float max_lse = std::max(p_lse, s_lse);

            const float p_lse_shifted = p_lse - max_lse;
            const float s_lse_shifted = s_lse - max_lse;

            const float p_exp = std::exp(p_lse_shifted); // prefix contribution
            const float s_exp = std::exp(s_lse_shifted); // suffix contribution
            const float out_se = p_exp + s_exp;          // normaliser

            // write merged log-sum-exp
            if (lse != nullptr) {
                lse[lse_idx] = std::log(out_se) + max_lse;
            }

            // compute the scales
            const float p_scale = p_exp / out_se;
            const float s_scale = s_exp / out_se;

            // write output
            const int base = t * num_heads * head_size + h * head_size;
            #pragma omp parallel for
            for (int d = 0; d < head_size; ++d) {
                float sum = to_float(prefix_output[base + d]) * p_scale +
                            to_float(suffix_output[base + d]) * s_scale;
                from_float(output[base + d], sum);
            }
        }
    }
}

template <typename scalar_t>
void merge_attn_states_kernel2(
          scalar_t* __restrict__ output,
    const scalar_t* __restrict__ prefix_output,
    const scalar_t* __restrict__ suffix_output,

          float* __restrict__ lse,
    const float* __restrict__ prefix_lse,
    const float* __restrict__ suffix_lse,

    const uint32_t num_tokens,
    const uint32_t num_heads,
    const uint32_t head_size)
{
  const uint32_t pack_size = 16 / sizeof(scalar_t);
  const uint32_t threads_per_head = head_size / pack_size;
  const uint32_t token_head_threads = num_tokens * num_heads * threads_per_head;
  const uint32_t NUM_THREADS = 128;
  const uint32_t grid = (token_head_threads + NUM_THREADS - 1) / NUM_THREADS;

  typedef struct alignas(16) {
    unsigned x; unsigned y; unsigned z; unsigned w;
  } pack_128b_t;

  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(grid) num_threads(NUM_THREADS)
  for (int bid = 0; bid < grid; ++bid) {
    for (int h = 0; h < NUM_THREADS; ++h) {
      const uint32_t global_idx = bid * NUM_THREADS + h;
      if (global_idx < token_head_threads) {
        const uint32_t token_head_idx = global_idx / threads_per_head;
        const uint32_t pack_idx = global_idx % threads_per_head;

        const uint32_t token_idx = token_head_idx / num_heads;
        const uint32_t head_idx = token_head_idx % num_heads;

        const uint32_t pack_offset = pack_idx * pack_size;

        // [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
        const uint32_t head_offset = token_idx * num_heads * head_size + head_idx * head_size;
        const scalar_t* prefix_head_ptr = prefix_output + head_offset;
        const scalar_t* suffix_head_ptr = suffix_output + head_offset;
              scalar_t* output_head_ptr = output + head_offset;

        // [NUM_TOKENS, NUM_HEADS]
        const uint32_t lse_offset = token_idx * num_heads + head_idx;
        float p_lse = prefix_lse[lse_offset];
        float s_lse = suffix_lse[lse_offset];
        p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
        s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

        const float max_lse = fmaxf(p_lse, s_lse);
        p_lse = p_lse - max_lse;
        s_lse = s_lse - max_lse;
        const float p_se = expf(p_lse);
        const float s_se = expf(s_lse);
        const float out_se = p_se + s_se;
        const float p_scale = p_se / out_se;
        const float s_scale = s_se / out_se;

        if (pack_offset < head_size) {
          pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_idx];
          pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_idx];
          pack_128b_t o_out_pack;

          #pragma unroll
          for (uint32_t i = 0; i < pack_size; ++i) {
            // Always use float for FMA to keep high precision.
            // half(uint32_t16_t), bfloat16, float -> float.
            const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
            const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);

            // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
            const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);

            // float -> half(uint32_t16_t), bfloat16, float.
            from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
          }

          reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_idx] = o_out_pack;
        }
        // We only need to write to lse once per head.
        if (lse != nullptr && pack_idx == 0) {
          float out_lse = logf(out_se) + max_lse;
          lse[lse_offset] = out_lse;
        }
      }
    }
  }
}

template <typename scalar_t>
void merge_attn_states_launcher(
   int32_t repeat,
  uint32_t num_tokens,
  uint32_t num_heads,
  uint32_t head_size
)
{
  uint64_t output_size = (uint64_t)num_tokens * num_heads * head_size;
  uint64_t output_size_bytes = (uint64_t)num_tokens * num_heads * head_size * sizeof(scalar_t);
  uint64_t lse_size = (uint64_t)num_tokens * num_heads;
  uint64_t lse_size_bytes = (uint64_t)num_tokens * num_heads * sizeof(float);

  scalar_t *h_prefix_output = (scalar_t*) malloc (output_size_bytes);
  scalar_t *h_suffix_output = (scalar_t*) malloc (output_size_bytes);
  scalar_t *h_output = (scalar_t*) malloc (output_size_bytes);
  scalar_t *r_output = (scalar_t*) malloc (output_size_bytes);

  float *h_prefix_lse = (float*) malloc (lse_size_bytes);
  float *h_suffix_lse = (float*) malloc (lse_size_bytes);
  float *h_lse = (float*) malloc (lse_size_bytes);
  float *r_lse = (float*) malloc (lse_size_bytes);

  #pragma omp target data map(alloc: h_prefix_output[0:output_size], \
                                     h_suffix_output[0:output_size], \
                                     h_output[0:output_size], \
                                     h_lse[0:lse_size], \
                                     h_prefix_lse[0:lse_size], \
                                     h_suffix_lse[0:lse_size])
  {
    const float scale = 1.f / std::sqrt((float)head_size);
    const unsigned long seed = 1234;
    uniform_fill_kernel(h_prefix_output, output_size, -scale, scale, seed);
    uniform_fill_kernel(h_suffix_output, output_size, -scale, scale, seed);
    uniform_fill_kernel(h_prefix_lse, lse_size, -scale, scale, seed);
    uniform_fill_kernel(h_suffix_lse, lse_size, -scale, scale, seed);

    #pragma omp target update from (h_prefix_output[0:output_size])
    #pragma omp target update from (h_suffix_output[0:output_size])
    #pragma omp target update from (h_prefix_lse[0:lse_size])
    #pragma omp target update from (h_suffix_lse[0:lse_size])

    reference<scalar_t>(
      r_output,
      h_prefix_output,
      h_suffix_output,
      r_lse,
      h_prefix_lse,
      h_suffix_lse,
      num_tokens,
      num_heads,
      head_size);

    // warmup
    for (int i = 0; i < 100; i++) {
      merge_attn_states_kernel<scalar_t>(
        h_output,
        h_prefix_output,
        h_suffix_output,
        h_lse,
        h_prefix_lse,
        h_suffix_lse,
        num_tokens,
        num_heads,
        head_size);
    }
    #pragma omp target update from (h_output[0:output_size])
    #pragma omp target update from (h_lse[0:lse_size])

    bool ok = true;
    for (uint64_t i = 0; i < output_size; i++) {
      if (std::fabs(to_float32(h_output[i]) - to_float32(r_output[i])) > 1e-3f) {
        ok = false;
        break;
      }
    }
    for (uint64_t i = 0; i < lse_size; i++) {
      if (std::fabs(to_float32(h_lse[i]) - to_float32(r_lse[i])) > 1e-3f) {
        ok = false;
        break;
      }
    }

    // warmup
    for (int i = 0; i < 100; i++) {
      merge_attn_states_kernel2<scalar_t>(
        h_output,
        h_prefix_output,
        h_suffix_output,
        h_lse,
        h_prefix_lse,
        h_suffix_lse,
        num_tokens,
        num_heads,
        head_size);
    }
    #pragma omp target update from (h_output[0:output_size])
    #pragma omp target update from (h_lse[0:lse_size])

    for (uint64_t i = 0; i < output_size; i++) {
      if (std::fabs(to_float32(h_output[i]) - to_float32(r_output[i])) > 1e-3f) {
        ok = false;
        break;
      }
    }
    for (uint64_t i = 0; i < lse_size; i++) {
      if (std::fabs(to_float32(h_lse[i]) - to_float32(r_lse[i])) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      merge_attn_states_kernel<scalar_t>(
        h_output,
        h_prefix_output,
        h_suffix_output,
        h_lse,
        h_prefix_lse,
        h_suffix_lse,
        num_tokens,
        num_heads,
        head_size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      merge_attn_states_kernel2<scalar_t>(
        h_output,
        h_prefix_output,
        h_suffix_output,
        h_lse,
        h_prefix_lse,
        h_suffix_lse,
        num_tokens,
        num_heads,
        head_size);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the kernel2: %f (us)\n", (time * 1e-3f) / repeat);
  }

  free(h_prefix_output);
  free(h_suffix_output);
  free(h_output);
  free(r_output);
  free(h_prefix_lse);
  free(h_suffix_lse);
  free(h_lse);
  free(r_lse);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of tokens> <number of heads> <head size> ", argv[0]);
    return 1;
  }
  uint32_t num_tokens = atoi(argv[1]);
  uint32_t num_heads = atoi(argv[2]);
  uint32_t head_size = atoi(argv[3]);
  int32_t repeat = atoi(argv[4]);

  printf("\n#tokens %d, #heads %d, head dimension %d\n", num_tokens, num_heads, head_size);
  printf("output dtype FP32: ");
  merge_attn_states_launcher<float>(repeat, num_tokens, num_heads, head_size);
  printf("output dtype FP16: ");
  merge_attn_states_launcher<_Float16>(repeat, num_tokens, num_heads, head_size);
  printf("----------------------------------------------------\n");

  return 0;
}
