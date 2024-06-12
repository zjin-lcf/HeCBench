#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "reference.h"

#define GPU_THREADS 256

#define KERNEL_LOOP(index, range) \
   for (int index = blockIdx.x * blockDim.x + threadIdx.x;  \
            index < (range); index += blockDim.x * gridDim.x) 

template <typename T>
__global__
void SwishKernel(const int N, const T* X, T* Y)
{
  KERNEL_LOOP(i, N) {
    Y[i] = __ldg(X + i) / (T(1) + exp(-__ldg(X + i)));
  }
}

template <typename T>
__global__
void SwishGradientKernel(
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
          T* dX)
{
  KERNEL_LOOP(i, N) {
    dX[i] = __ldg(dY + i) *
            (__ldg(Y + i) + (T(1) - __ldg(Y + i)) / (T(1) + exp(-__ldg(X + i))));
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

  T *d_X, *d_Y, *d_dX, *d_dY;
  hipMalloc((void**)&d_X, size_bytes);
  hipMemcpy(d_X, h_X, size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_Y, size_bytes);

  hipMalloc((void**)&d_dY, size_bytes);
  hipMemcpy(d_dY, h_dY, size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_dX, size_bytes);

  dim3 grid ((N + GPU_THREADS - 1) / GPU_THREADS);
  dim3 block (GPU_THREADS);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    SwishKernel <<<grid, block>>> (N, d_X, d_Y);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Swish kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) 
    SwishGradientKernel <<<grid, block>>> (N, d_X, d_Y, d_dY, d_dX);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SwishGradient kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  hipMemcpy(h_dX, d_dX, size_bytes, hipMemcpyDeviceToHost); 
  hipMemcpy(h_Y, d_Y, size_bytes, hipMemcpyDeviceToHost); 

  reference (N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_dX[i] - r_dX[i]) > 1e-3 || fabs(h_Y[i] - r_Y[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  hipFree(d_X);
  hipFree(d_Y);
  hipFree(d_dX);
  hipFree(d_dY);

  free(h_X);
  free(h_Y);
  free(h_dX);
  free(h_dY);
  free(r_dX);
  free(r_Y);
  if (!ok) exit(1);
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
