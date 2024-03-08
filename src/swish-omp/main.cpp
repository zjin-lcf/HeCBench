#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include "reference.h"

#define GPU_THREADS 256

template <typename T>
void SwishKernel(const int N, const T* X, T* Y)
{
  #pragma omp target teams distribute parallel for num_threads(GPU_THREADS)
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] / (T(1) + exp(-X[i]));
  }
}

template <typename T>
void SwishGradientKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
          T* dX)
{
  #pragma omp target teams distribute parallel for num_threads(GPU_THREADS)
  for (int i = 0; i < N; i++) {
    dX[i] = dY[i] * (Y[i] + (T(1) - Y[i]) / (T(1) + exp(-X[i])));
  }
}

template<typename T>
void eval_swish (const int N, const int repeat) {

  size_t size_bytes = N * sizeof(T); 

  T *h_X  = (T*) malloc (size_bytes);
  T *h_Y  = (T*) malloc (size_bytes);
  T *h_dY = (T*) malloc (size_bytes);
  T *h_dX = (T*) malloc (size_bytes);
  T *r_Y  = (T*) malloc (size_bytes);
  T *r_dX = (T*) malloc (size_bytes);

  std::default_random_engine gen (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < N; i++) {
    h_X[i] = distr(gen);
    h_dY[i] = distr(gen);
  }

  #pragma omp target data map(to: h_X[0:N], h_dY[0:N]) \
                          map(from: h_Y[0:N], h_dX[0:N]) 
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) 
      SwishKernel(N, h_X, h_Y);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of Swish kernel: %f (us)\n", (time * 1e-3f) / repeat);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) 
      SwishGradientKernel(N, h_X, h_Y, h_dY, h_dX);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SwishGradient kernel: %f (us)\n", (time * 1e-3f) / repeat);
  }

  // verify
  reference (N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_dX[i] - r_dX[i]) > 1e-3 || fabs(h_Y[i] - r_Y[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(h_X);
  free(h_Y);
  free(h_dX);
  free(h_dY);
  free(r_dX);
  free(r_Y);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }

  const int N = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval_swish<float>(N, repeat);

  return 0;
}
