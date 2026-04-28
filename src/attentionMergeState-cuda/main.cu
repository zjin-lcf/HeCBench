#include <chrono>
#include <cstdio>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reference.h"

#define GPU_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", \
               __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
} while (0)

__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

__device__ __forceinline__ float uint32_to_uniform(uint32_t x) {
    // Convert to (0,1)
    return (x >> 8) * 0x1.0p-24f;
}

template <typename T>
__global__ void uniform_fill_kernel(
    T* data,
    uint64_t n,
    float low,
    float high,
    uint32_t seed
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint32_t rng = seed ^ idx;

    uint32_t r = xorshift32(rng);
    float u = uint32_to_uniform(r);
    float v = low + (high - low) * u;
    data[idx] = (T)v;
}

// Helper functions to convert between different data types
// (float, half, bfloat16) for the merge attention states kernel.
inline __device__ float to_float(float u) {
  return u;
}
inline __device__ float to_float(half u) {
  return __half2float(u);
}
inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}
inline __device__ void from_float(float& d, float s) {
  d = s;
}
inline __device__ void from_float(half& d, float s) {
  d = __float2half(s);
}
inline __device__ void from_float(__nv_bfloat16& d, float s) {
  d = __float2bfloat16(s);
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
__global__ void merge_attn_states_kernel(
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
  // recompute pack size, threads_per_head and token_head_threads
  // for kernel performance..
  using pack_128b_t = uint4;
  const uint32_t pack_size = 16 / sizeof(scalar_t);
  const uint32_t threads_per_head = head_size / pack_size;

  const uint32_t global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
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
    float out_lse = logf(out_se) + max_lse;
    lse[lse_offset] = out_lse;
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

  std::vector<scalar_t> h_prefix_output(output_size), h_suffix_output(output_size),
                        h_output(output_size), r_output(output_size);
  std::vector<float> h_prefix_lse(lse_size), h_suffix_lse(lse_size),
                     h_lse(lse_size), r_lse(lse_size);

  scalar_t *d_prefix_output, *d_suffix_output, *d_output;
  float *d_prefix_lse, *d_suffix_lse, *d_lse;

  GPU_CHECK(cudaMalloc(&d_prefix_output, output_size_bytes));
  GPU_CHECK(cudaMalloc(&d_suffix_output, output_size_bytes));
  GPU_CHECK(cudaMalloc(&d_output, output_size_bytes));
  GPU_CHECK(cudaMalloc(&d_prefix_lse, lse_size_bytes));
  GPU_CHECK(cudaMalloc(&d_suffix_lse, lse_size_bytes));
  GPU_CHECK(cudaMalloc(&d_lse, lse_size_bytes));

  const int threads = 256;
  const float scale = 1.f / std::sqrt((float)head_size);
  const unsigned long seed = 1234;
  int blocks = (output_size + threads - 1) / threads;
  uniform_fill_kernel<<<blocks, threads>>>(d_prefix_output, output_size, -scale, scale, seed);
  uniform_fill_kernel<<<blocks, threads>>>(d_suffix_output, output_size, -scale, scale, seed);

  blocks = (lse_size + threads - 1) / threads;
  uniform_fill_kernel<<<blocks, threads>>>(d_prefix_lse, lse_size, -scale, scale, seed);
  uniform_fill_kernel<<<blocks, threads>>>(d_suffix_lse, lse_size, -scale, scale, seed);

  GPU_CHECK(cudaMemcpy(h_prefix_output.data(), d_prefix_output, output_size_bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(h_suffix_output.data(), d_suffix_output, output_size_bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(h_prefix_lse.data(), d_prefix_lse, lse_size_bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(h_suffix_lse.data(), d_suffix_lse, lse_size_bytes, cudaMemcpyDeviceToHost));

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
  dim3 block(NUM_THREADS);
  dim3 grid((total_threads + NUM_THREADS - 1) / NUM_THREADS);

  // warmup
  for (int i = 0; i < 100; i++) {
    merge_attn_states_kernel<scalar_t, NUM_THREADS><<<grid, block>>>(
      d_output,
      d_prefix_output,
      d_suffix_output,
      d_lse,
      d_prefix_lse,
      d_suffix_lse,
      num_tokens,
      num_heads,
      head_size);
  }
  GPU_CHECK(cudaMemcpy(h_output.data(), d_output, output_size_bytes, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(h_lse.data(), d_lse, lse_size_bytes, cudaMemcpyDeviceToHost));

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


  GPU_CHECK(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    merge_attn_states_kernel<scalar_t, NUM_THREADS><<<grid, block>>>(
      d_output,
      d_prefix_output,
      d_suffix_output,
      d_lse,
      d_prefix_lse,
      d_suffix_lse,
      num_tokens,
      num_heads,
      head_size);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  GPU_CHECK(cudaFree(d_prefix_output));
  GPU_CHECK(cudaFree(d_suffix_output));
  GPU_CHECK(cudaFree(d_output));
  GPU_CHECK(cudaFree(d_prefix_lse));
  GPU_CHECK(cudaFree(d_suffix_lse));
  GPU_CHECK(cudaFree(d_lse));
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
  merge_attn_states_launcher<__half>(repeat, num_tokens, num_heads, head_size);
  printf("output dtype BF16: ");
  merge_attn_states_launcher<__nv_bfloat16>(repeat, num_tokens, num_heads, head_size);
  printf("----------------------------------------------------\n");

  return 0;
}
