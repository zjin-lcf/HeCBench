#include <chrono>
#include <cstdio>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

inline uint32_t xorshift32(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

inline float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}

template <typename T>
void uniform_fill_kernel(
    T* data,
    uint64_t n,
    float low,
    float high,
    uint32_t seed,
    sycl::nd_item<1> &item)
{
    uint64_t idx = item.get_global_id(0);
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
}

// Helper functions to convert between different data types
// (float, half, bfloat16) for the merge attention states kernel.
template <typename T>
inline float to_float(T u) {
  return static_cast<float>(u);
}

template <typename T>
inline void from_float(T& d, float s) {
  d = static_cast<T>(s);
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
template <typename scalar_t, const uint32_t NUM_THREADS>
void merge_attn_states_kernel(scalar_t *__restrict__ output,
                              const scalar_t *__restrict__ prefix_output,
                              const scalar_t *__restrict__ suffix_output,

                              float *__restrict__ lse,
                              const float *__restrict__ prefix_lse,
                              const float *__restrict__ suffix_lse,

                              const uint32_t num_tokens,
                              const uint32_t num_heads,
                              const uint32_t head_size,
                              sycl::nd_item<1> &item)
{
  // recompute pack size, threads_per_head and token_head_threads
  // for kernel performance..
  using pack_128b_t = sycl::uint4;
  const uint32_t pack_size = 16 / sizeof(scalar_t);
  const uint32_t threads_per_head = head_size / pack_size;

  const uint32_t global_idx = item.get_group(0) * NUM_THREADS + item.get_local_id(0);
  const uint32_t token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
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
  p_lse = sycl::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = sycl::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = sycl::fmax(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = sycl::native::exp(p_lse);
  const float s_se = sycl::native::exp(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  // pack_offset / pack_size = pack_idx;
  if (pack_offset < head_size) {
    // Pack 128b load
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

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_idx] = o_out_pack;
  }
  // We only need to write to lse once per head.
  if (lse != nullptr && pack_idx == 0) {
    float out_lse = sycl::log((float)out_se) + max_lse;
    lse[lse_offset] = out_lse;
  }
}

template <typename scalar_t>
void merge_attn_states_launcher(sycl::queue &q, int32_t repeat, uint32_t num_tokens,
                                uint32_t num_heads, uint32_t head_size) try {
  uint64_t output_size = (uint64_t)num_tokens * num_heads * head_size;
  uint64_t output_size_bytes = (uint64_t)num_tokens * num_heads * head_size * sizeof(scalar_t);
  uint64_t lse_size = (uint64_t)num_tokens * num_heads;
  uint64_t lse_size_bytes = (uint64_t)num_tokens * num_heads * sizeof(float);

  std::vector<scalar_t> h_prefix_output(output_size), h_suffix_output(output_size),
                        h_output(output_size), r_output(output_size);
  std::vector<float> h_prefix_lse(lse_size), h_suffix_lse(lse_size),
                     h_lse(lse_size), r_lse(lse_size);

  scalar_t *d_prefix_output, *d_suffix_output, *d_output;
  float *d_prefix_lse, *d_suffix_lse, *d_lse;

  d_prefix_output = sycl::malloc_device<scalar_t>(output_size, q);
  d_suffix_output = sycl::malloc_device<scalar_t>(output_size, q);
  d_output = sycl::malloc_device<scalar_t>(output_size, q);
  
  d_prefix_lse = sycl::malloc_device<float>(lse_size, q);
  d_suffix_lse = sycl::malloc_device<float>(lse_size, q);
  d_lse = sycl::malloc_device<float>(lse_size, q);

  const int threads = 256;
  const float scale = 1.f / std::sqrt((float)head_size);
  const unsigned long seed = 1234;
  int blocks = (output_size + threads - 1) / threads;

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads),
                                       sycl::range<1>(threads)),
      [=](sycl::nd_item<1> item) {
        uniform_fill_kernel(d_prefix_output, output_size,
                            -scale, scale, seed, item);
    });
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads),
                                       sycl::range<1>(threads)),
      [=](sycl::nd_item<1> item) {
        uniform_fill_kernel(d_suffix_output, output_size,
                            -scale, scale, seed, item);
    });
  });

  blocks = (lse_size + threads - 1) / threads;
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads),
                                       sycl::range<1>(threads)),
      [=](sycl::nd_item<1> item) {
        uniform_fill_kernel(d_prefix_lse, lse_size,
                            -scale, scale, seed, item);
    });
  });
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads),
                                       sycl::range<1>(threads)),
      [=](sycl::nd_item<1> item) {
        uniform_fill_kernel(d_suffix_lse, lse_size,
                            -scale, scale, seed, item);
    });
  });
  
  q.memcpy(h_prefix_output.data(), d_prefix_output, output_size_bytes);
  q.memcpy(h_suffix_output.data(), d_suffix_output, output_size_bytes);
  q.memcpy(h_prefix_lse.data(), d_prefix_lse, lse_size_bytes);
  q.memcpy(h_suffix_lse.data(), d_suffix_lse, lse_size_bytes);
  q.wait();

  reference<scalar_t>(
    r_output.data(),
    h_prefix_output.data(),
    h_suffix_output.data(),
    r_lse.data(),
    h_prefix_lse.data(),
    h_suffix_lse.data(),
    num_tokens,
    num_heads,
    head_size);

  const uint32_t pack_size = 16 / sizeof(scalar_t);
  if (head_size % pack_size != 0) {
    printf("headsize must be multiple of pack_size %d\n", pack_size);
    return;
  }
  // Process one pack elements per thread. for float, the
  // pack_size is 4 for half/bf16, the pack_size is 8.
  const uint32_t threads_per_head = head_size / pack_size;
  const uint32_t total_threads = num_tokens * num_heads * threads_per_head;

  constexpr uint32_t NUM_THREADS = 128;
  sycl::range<1> lws (NUM_THREADS);
  sycl::range<1> gws ((total_threads + NUM_THREADS - 1) / NUM_THREADS * NUM_THREADS);

  auto kFn = [&](sycl::handler& cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        merge_attn_states_kernel<scalar_t, NUM_THREADS>(
            d_output, d_prefix_output, d_suffix_output, d_lse, d_prefix_lse,
            d_suffix_lse, num_tokens, num_heads, head_size, item);
      });
  };

  // warmup
  for (int i = 0; i < 100; i++) {
    q.submit(kFn);
  }
  
  q.memcpy(h_output.data(), d_output, output_size_bytes);
  q.memcpy(h_lse.data(), d_lse, lse_size_bytes).wait();
  q.wait();

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
  printf("%s\n", ok ? "PASS" : "FAIL");

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(kFn);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_prefix_output, q);
  sycl::free(d_suffix_output, q);
  sycl::free(d_output, q);
  sycl::free(d_prefix_lse, q);
  sycl::free(d_suffix_lse, q);
  sycl::free(d_lse, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("\n#tokens %d, #heads %d, head dimension %d\n", num_tokens, num_heads, head_size);
  printf("output dtype FP32: ");
  merge_attn_states_launcher<float>(q, repeat, num_tokens, num_heads, head_size);
  printf("output dtype FP16: ");
  merge_attn_states_launcher<sycl::half>(q, repeat, num_tokens, num_heads,
                                         head_size);
  printf("output dtype BF16: ");
  merge_attn_states_launcher<sycl::ext::oneapi::bfloat16>(q, repeat, num_tokens,
                                                          num_heads, head_size);
  printf("----------------------------------------------------\n");

  return 0;
}
