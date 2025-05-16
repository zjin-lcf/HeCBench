#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

#define NUM_THREADS 256
#define GridDimMaxY 65536

#ifdef V2

template <typename T>
void ChannelShuffleNCHWKernel_opt(
    const int numTeams,
    const int numThreads,
    const int N,
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  const int C = G * K;
  #pragma omp target teams distribute parallel for collapse(3) \
   num_teams(numTeams) num_threads(numThreads)
  for (int n = 0; n < N; n++)
    for (int c = 0; c < C; c++)
      for (int s = 0; s < HxW; s++)
        Y[(n * C + c) * HxW + s] = X[(n * C + (c % G) * K + c / G) * HxW + s];
}

template <typename T>
void ChannelShuffleNHWCKernel_opt(
    const int numTeams,
    const int numThreads,
    const int O,
    const int G,
    const int K,
    const T* X,
          T* Y)
{
  const int C = G * K;
  #pragma omp target teams distribute parallel for collapse(2) \
   num_teams(numTeams) num_threads(numThreads)
  for (int o = 0; o < O; o++)
    for (int i = 0; i < C; i++)
      Y[o*C + i] = X[o*C + (i % G) * K + i / G];
}

#else

// begin of ChannelShuffleNHWCKernel
template <typename T, int kSharedSize>
void ChannelShuffleNHWCKernel(
    const int numTeams,
    const int numThreads,
    const int G,
    const int K,
    const T* X,
          T* Y)
{
  #pragma omp target teams num_teams(numTeams) 
  {
    T sdata[kSharedSize];
    #pragma omp parallel num_threads(numThreads)
    {
      const int C = G * K;
      const int offset = omp_get_team_num() * C;
      for (int i = omp_get_thread_num(); i < C; i += omp_get_num_threads()) {
        sdata[i] = X[offset + i];
      }
      #pragma omp barrier
      for (int i = omp_get_thread_num(); i < C; i += omp_get_num_threads()) {
        const int g = i % G;
        const int k = i / G;
        Y[offset + i] = sdata[g * K + k];
      }
    }
  }
}
// end of ChannelShuffleNHWCKernel

// begin of ChannelShuffleNCHWKernel
template <typename T, bool kNFirst >
void ChannelShuffleNCHWKernel (
    const int numTeams,
    const int numThreads,
    const int S,
    const int N,
    const int C,
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  #pragma omp target teams num_teams(numTeams) 
  {
    #pragma omp parallel num_threads(numThreads)
    {
      const int blockIdx_x = omp_get_team_num() % S;
      const int blockIdx_y = omp_get_team_num() / S % N;
      const int blockIdx_z = omp_get_team_num() / (S * N);
      const int n = kNFirst ? blockIdx_x : blockIdx_y;
      const int s = kNFirst ? blockIdx_y : blockIdx_x;
      const int g = blockIdx_z % G ;
      const int k = blockIdx_z / G ;
      const int offset = s * NUM_THREADS + omp_get_thread_num();
      if (offset < HxW) {
        Y[(n * C + blockIdx_z) * HxW + offset] =
            X[(n * C + g * K + k) * HxW + offset];
      }
    }
  }
}
// end of ChannelShuffleNCHWKernel

#endif

template <typename T>
bool ChannelShuffleNCHW (T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  const int numTeams = S * N * C; 
  const int numThreads = NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

#ifdef V2
  for (int i = 0; i < repeat; i++) {
    ChannelShuffleNCHWKernel_opt<float>(numTeams, numThreads, N, G, K, HxW, X, Y);
  }
#else
  if (N <= GridDimMaxY) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNCHWKernel<float, false>(numTeams, numThreads, S, N, C, G, K, HxW, X, Y);
  } else {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNCHWKernel<float, true>(numTeams, numThreads, N, S, C, G, K, HxW, X, Y);
  }
#endif

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
  const int numTeams = O;
  const int numThreads = NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

#ifdef V2
  for (int i = 0; i < repeat; i++) {
    ChannelShuffleNHWCKernel_opt<float>(numTeams, numThreads, O, G, K, X, Y);
  }
#else
  if (C <= 32) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 32>(numTeams, numThreads, G, K, X, Y);
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 128>(numTeams, numThreads, G, K, X, Y);
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++)
      ChannelShuffleNHWCKernel<float, 512>(numTeams, numThreads, G, K, X, Y);
  }
#endif

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
