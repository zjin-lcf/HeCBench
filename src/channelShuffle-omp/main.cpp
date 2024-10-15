#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256

template <typename T>
void ChannelShuffleNCHWKernel(
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  const int C = G * K;
  #pragma omp target teams distribute parallel for collapse(3) num_threads(NUM_THREADS)
  for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
      for (int s = 0; s < HxW; s++)
        Y[(n * C + c) * HxW + s] = X[(n * C + (c % G) * K + c / G) * HxW + s];
}

template <typename T>
void
ChannelShuffleNHWCKernel(const int O, const int G, const int K, const T* X, T* Y)
{
  const int C = G * K;
  #pragma omp target teams distribute parallel for collapse(2) num_threads(NUM_THREADS)
  for (int o = 0; o < O; o++)
    for (int i = 0; i < C; i++)
      Y[o*C + i] = X[o*C + (i % G) * K + i / G];
}

template <typename T>
bool ChannelShuffleNCHW (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelShuffleNCHWKernel<float>(N, G, K, HxW, X, Y);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int O = N * HxW;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelShuffleNHWCKernel<float>(O, G, K, X, Y);
  }

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <group size> <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int G = atoi(argv[1]);
  const int W = atoi(argv[2]);
  const int H = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  long time;
  float *h_X, *h_Y, *h_Y_ref;
  int error;

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      const int numel = N * C * W * H; // assume no integer overflow
      size_t data_size_bytes = numel * sizeof(float);

      h_X = (float*) malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float) i / numel;

      h_Y = (float*) malloc(data_size_bytes);
      h_Y_ref = (float*) malloc(data_size_bytes);

      #pragma omp target data map(to: h_X[0:numel]) map(alloc: h_Y[0:numel])
      {
        ChannelShuffleNHWC (h_X, N, C, G, numel, h_Y, time, repeat);
        #pragma omp target update from (h_Y[0:numel])
        ChannelShuffleNHWC_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
        error = memcmp(h_Y, h_Y_ref, data_size_bytes);
        if (error)
          printf("Failed to pass channel shuffle (NHWC) check\n");
        else
          printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

        ChannelShuffleNCHW (h_X, N, C, G, numel, h_Y, time, repeat);
        #pragma omp target update from (h_Y[0:numel])
        ChannelShuffleNCHW_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
        error = memcmp(h_Y, h_Y_ref, data_size_bytes);
        if (error)
          printf("Failed to pass channel shuffle (NCHW) check\n");
        else
          printf("Average time of channel shuffle (NCHW): %f (ms)\n", (time * 1e-6f) / repeat);
      }

      free(h_X);
      free(h_Y);
      free(h_Y_ref);
    }
  }

  return 0;
}
