#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <omp.h>
#include "kernels.h"
#include "reference.h"

template <typename scalar_t>
int64_t moe_sum(const scalar_t* input,   // [num_tokens, topk, hidden_size]
                      scalar_t* output,  // [num_tokens, hidden_size]
                const int hidden_size,
                const int num_tokens,
                const int topk,
                const int repeat)
{
  int block_size = std::min(hidden_size, 1024);

  // warmup
  for (int i = 0; i < 100; i++) {
    moe_sum_kernel<scalar_t, 2>(
        output, input, hidden_size, num_tokens, block_size);
  }

  auto start = std::chrono::steady_clock::now();

  switch (topk) {
    case 2:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel<scalar_t, 2>(output, input, hidden_size, num_tokens, block_size);
      break;

    case 3:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel<scalar_t, 3>(output, input, hidden_size, num_tokens, block_size);
      break;

    case 4:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel<scalar_t, 4>(output, input, hidden_size, num_tokens, block_size);
      break;

    default:
      printf("Not implemented for topK value %d\n", topk);
      break;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel (TopK = %d): %f (us)\n",
         topk, (time * 1e-3f) / repeat);
  return time;
}

template <typename scalar_t>
int64_t moe_sum_vec(const scalar_t* input,
                          scalar_t* output,
                    const int hidden_size,
                    const int num_tokens,
                    const int topk,
                    const int repeat)
{
  int block_size = std::min(hidden_size/4, 1024);

  // warmup
  for (int i = 0; i < 100; i++) {
    moe_sum_kernel_vec<scalar_t, 2>(output, input, hidden_size, num_tokens, block_size);
  }

  auto start = std::chrono::steady_clock::now();

  switch (topk) {
    case 2:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel_vec<scalar_t, 2>(output, input, hidden_size, num_tokens, block_size);
      break;

    case 3:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel_vec<scalar_t, 3>(output, input, hidden_size, num_tokens, block_size);
      break;

    case 4:
      for (int i = 0; i < repeat; i++)
        moe_sum_kernel_vec<scalar_t, 4>(output, input, hidden_size, num_tokens, block_size);
      break;

    default:
      printf("Not implemented for topK value %d\n", topk);
      break;
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of vec4 kernel (TopK = %d): %f (us)\n",
         topk, (time * 1e-3f) / repeat);
  return time;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of tokens> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int32_t num_tokens = atoi(argv[1]);
  const int32_t hidden_size = atoi(argv[2]);
  const int32_t repeat = atoi(argv[3]);

  if (hidden_size % 4) {
    printf("Hidden size is a multiple of four\n");
    return 1;
  }

  const int64_t output_size = (int64_t)num_tokens * hidden_size;
  const int64_t output_size_bytes = output_size * sizeof(float);
  float *output, *output_vec;
  float *r_output;

  r_output = (float*) malloc (output_size_bytes);
  output = (float*) malloc (output_size_bytes);
  output_vec = (float*) malloc (output_size_bytes);

  #pragma omp target enter data map(alloc: output[0:output_size], output_vec[0:output_size])

  for (int topk = 2; topk <= 4; topk++) {
    const int64_t input_size = output_size * topk;
    const int64_t input_size_bytes = input_size * sizeof(float);

    float *input;
    input = (float*) malloc (input_size_bytes);
    #pragma omp target enter data map(to: input[0:input_size])

    std::mt19937 gen(topk);
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int64_t i = 0; i < (int64_t)num_tokens * hidden_size * topk; i++) {
      input[i] = dis(gen);
    }

    // reference
    moe_sum_ref<float>(topk, r_output, input, num_tokens, hidden_size);

    #pragma omp target update to(input[0:input_size])

    // base
    int64_t nano_seconds = moe_sum(input, output, hidden_size, num_tokens, topk, repeat);

    #pragma omp target update from(output[0:output_size])

    bool ok = true;
    for (int64_t i = 0; i < (int64_t)num_tokens * hidden_size; i++) {
      if (fabsf(r_output[i] - output[i]) > 1e-4f) {
        ok = false; break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    float io_bytes = 1.f * repeat * (input_size_bytes + output_size_bytes);
    float bw = io_bytes / nano_seconds;
    printf("Kernel bandwidth: %f GB/s \n", bw);

    // vectorized
    nano_seconds = moe_sum_vec(input, output_vec, hidden_size, num_tokens, topk, repeat);
    #pragma omp target update from(output_vec[0:output_size])
    int32_t rc = memcmp(output, output_vec, output_size_bytes);
    printf("%s\n", rc ? "FAIL" : "PASS");

    float bw_vec4 = io_bytes / nano_seconds;
    printf("Kernel(vec4) bandwidth: %f GB/s (%f%%)\n", bw_vec4, 100 * (bw_vec4 - bw) / bw);

    #pragma omp target exit data map(delete: input[0:input_size])
    free(input);
  }

  #pragma omp target exit data map(delete: output[0:output_size])
  #pragma omp target exit data map(delete: output_vec[0:output_size])
  free(r_output);
  free(output);
  free(output_vec);
  return 0;
}
