#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

// begin of attention_kernel1
void attention_kernel1 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ key,
    const float*__restrict__ query,
    float*__restrict__ dot_product,
    float*__restrict__ exp_sum,
    const int n,
    const int d)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
   for (int i = 0; i < n; i++) {
     float sum = 0;
     for (int j = 0; j < d; j++)
        sum += key[i * d + j] * query[j];
     dot_product[i] = sum;
     #pragma omp atomic update  
     exp_sum[0] += expf(sum);
   }
}
// end of attention_kernel1

// begin of attention_kernel2
void attention_kernel2 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ exp_sum,
    const float*__restrict__ dot_product,
    float*__restrict__ score,
    const int n)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
   for (int i = 0; i < n; i++)
     score[i] = expf(dot_product[i]) / exp_sum[0];
}
// end of attention_kernel2

// begin of attention_kernel3
void attention_kernel3 (
    const int numTeams,
    const int numThreads,
    const float*__restrict__ score,
    const float*__restrict__ value,
    float*__restrict__ output,
    const int n,
    const int d)
{
  #pragma omp target teams distribute parallel for \
   num_teams(numTeams) num_threads(numThreads)
  for (int j = 0; j < d; j++) {
    float sum = 0;
    for (int i = 0; i < n; i++)
       sum += score[i] * value[i * d + j];
    output[j] = sum;
  }
}
// end of attention_kernel3

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat) 
{
  size_t kv_size = (size_t)d * n;
  // intermediate
  float* dot_product = (float*) malloc (n * sizeof(float));
  float* score = (float*) malloc (n * sizeof(float));
  float* exp_sum = (float*) malloc (sizeof(float));

  // result
  float* output = (float*) malloc (d * sizeof(float));

  #pragma omp target data map(to: key[0:kv_size], value[0:kv_size], query[0:d]), \
                          map(alloc: dot_product[0:n], score[0:n], exp_sum[0:1]), \
                          map(from: output[0:d])
  {
    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      exp_sum[0] = 0;
      #pragma omp target update to (exp_sum[0:1])

      attention_kernel1((n+255)/256, 256, key, query, dot_product, exp_sum, n, d);

      attention_kernel2((n+255)/256, 256, exp_sum, dot_product, score, n);
      
      attention_kernel3((d+255)/256, 256, score, value, output, n, d);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  free(dot_product);
  free(score);
  free(exp_sum);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  size_t q_size = (size_t)d;
  size_t kv_size = (size_t)d * n;
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  float* key = (float*) malloc (kv_size_bytes);
  float* value = (float*) malloc (kv_size_bytes);
  float* query = (float*) malloc (q_size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (size_t i = 0; i < kv_size; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);

  float* dout = attention_device(key, value, query, n, d, k, r);

  bool ok = true;
  for (int i = 0; i < d; i++) {
    if (fabsf(hout[i] - dout[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
