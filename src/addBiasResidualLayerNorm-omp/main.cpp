#include <algorithm>
#include <chrono>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "kernels.h"
#include "reference.h"


// Wrapper
template<typename T>
void invokeAddBiasResidualLayerNorm_omp(
          T* out,
    const T* input,
    const T* bias,
    const T* gamma,
    const T* beta,
    float layernorm_eps,
    int m,
    int n,
    int block_size)
{
  generalAddBiasResidualPostLayerNorm_omp<T>(
    out, input, bias, gamma, beta, layernorm_eps, m, n, block_size);
}

template<typename T, int V>
void layer(int m, int n, int repeat) {

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dis(0.f, 1.f);

  const int size = m * n;

  T* h_input  = (T*) malloc(size * sizeof(T));
  T* h_output = (T*) malloc(size * sizeof(T));
  T* h_bias   = (T*) malloc(n * sizeof(T));
  T* h_gamma  = (T*) malloc(n * sizeof(T));
  T* h_beta   = (T*) malloc(n * sizeof(T));

  for (int i = 0; i < size; i++)
    h_input[i] = (T) dis(gen);

  for (int i = 0; i < n; i++) {
    h_bias[i]  = (T) dis(gen);
    h_gamma[i] = (T) dis(gen);
    h_beta[i]  = (T) dis(gen);
  }

  float eps = 1e-6f;
  int block_size = std::min(n, 256);

  #pragma omp target data map(to: h_input[0:size], h_bias[0:n], h_gamma[0:n], h_beta[0:n]) \
                          map(tofrom: h_output[0:size])
  {
    // initialize output
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < size; i++) {
      h_output[i] = (T)0;
    }

    T *r_output = (T*) calloc (size, sizeof(T));
    for (int i = 0; i < 100; i++) {
      invokeAddBiasResidualLayerNorm_omp<T>(
          h_output, h_input, h_bias, h_gamma, h_beta, eps, m, n, block_size);
      reference<T>(r_output, h_input, h_bias, h_gamma, h_beta, eps, m, n);
    }
    #pragma omp target update from (h_output[0:size])

    bool ok = true;
    float error_bound = sizeof(T) >= 4 ? 1e-4f : 0.5f;
    for (int i = 0; i < size; i++) {
      if (fabsf(float(h_output[i]) - float(r_output[i])) > error_bound) {
        printf("i=%d %f != %f (ref)\n", i, float(h_output[i]), float(r_output[i]));
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    free(r_output);

    auto start = std::chrono::steady_clock::now();

    for (int r = 0; r < repeat; r++) {
      invokeAddBiasResidualLayerNorm_omp<T>(
          h_output, h_input, h_bias, h_gamma, h_beta, eps, m, n, block_size);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    printf("Average execution time (%d x %d): %f us\n",
           m, n, (time * 1e-3f) / repeat);
  }

  free(h_input);
  free(h_output);
  free(h_bias);
  free(h_gamma);
  free(h_beta);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  int repeat = atoi(argv[1]);

  const int m = 4096;  // batch size

  // The minimum value of n is 256
  for (int n = 256; n <= 8192; n = n * 2) {
    printf("---------------- float32 (version 1) -------------\n");
    layer<float, 1>(m, n, repeat);
    //printf("---------------- float32 (version 2) -------------\n");
    //layer<float, 2>(m, n, repeat);

    printf("---------------- float16 (version 1) -------------\n");
    layer<_Float16, 1>(m, n, repeat);
    //printf("---------------- float16 (version 2) -------------\n");
    //layer<_Float16, 2>(m, n, repeat);

    // not all openmp compiler support the bfloat16 type
    //printf("---------------- bfloat16 (version 1) -------------\n");
    //layer<__bf16, 1>(m, n, repeat);
    //printf("---------------- bfloat16 (version 2) -------------\n");
    //layer<_bf16, 2>(m, n, repeat);
    printf("\n");
  }
  return 0;
}
