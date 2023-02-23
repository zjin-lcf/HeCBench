#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"

#define NUM_THREADS 256
#define GridDimMaxY 65536

template <typename T, bool kNFirst>
void ChannelShuffleNCHWKernel(
    nd_item<3> &item,
    const int G,
    const int K,
    const int HxW,
    const T* X,
          T* Y)
{
  int blockIdx_x  = item.get_group(2);
  int threadIdx_x = item.get_local_id(2);
  int blockIdx_y  = item.get_group(1);
  int blockIdx_z  = item.get_group(0);

  const int C = G * K;
  const int n = kNFirst ? blockIdx_x : blockIdx_y;
  const int s = kNFirst ? blockIdx_y : blockIdx_x;
  const int g = blockIdx_z % G;
  const int k = blockIdx_z / G;
  const int offset = s * NUM_THREADS + threadIdx_x;
  if (offset < HxW) {
    Y[(n * C + blockIdx_z) * HxW + offset] =
        X[(n * C + g * K + k) * HxW + offset];
  }
}

template <typename T, int kSharedSize>
void ChannelShuffleNHWCKernel(
    nd_item<1> &item,
    const int G,
    const int K,
    const T* X,
          T* Y)
{
  int blockIdx_x  = item.get_group(0);
  int blockDim_x  = item.get_local_range(0);
  int threadIdx_x = item.get_local_id(0);

  auto g = item.get_group();
  multi_ptr<T[kSharedSize], access::address_space::local_space> localPtr =
    ext::oneapi::group_local_memory_for_overwrite<T[kSharedSize]>(g);

  T* sdata = *localPtr;

  const int C = G * K;
  const int offset = blockIdx_x * C;
  for (int i = threadIdx_x; i < C; i += blockDim_x) {
    sdata[i] = X[offset + i];
  }

  group_barrier(g, memory_scope::work_group);

  for (int i = threadIdx_x; i < C; i += blockDim_x) {
    const int g = i % G;
    const int k = i / G;
    Y[offset + i] = sdata[g * K + k];
  }
}

template <typename T>
bool ChannelShuffleNCHW (queue &q, T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false; 

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  if (N <= GridDimMaxY) {
    range<3> gws (C, N, S * NUM_THREADS);
    range<3> lws (1, 1, NUM_THREADS);

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class shuffle_nchw>(
          nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          ChannelShuffleNCHWKernel<float, false>(item, G, K, HxW, X, Y);
        });
      });
    }
  } else {
    range<3> gws (C, S, N * NUM_THREADS);
    range<3> lws (1, 1, NUM_THREADS);

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class shuffle2_nchw>(
          nd_range<3>(gws, lws), [=] (nd_item<3> item) {
          ChannelShuffleNCHWKernel<float, true>(item, G, K, HxW, X, Y);
        });
      });
    }
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return true;
}

template <typename T>
bool ChannelShuffleNHWC (queue &q, T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false; 

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int outer_size = N * HxW;

  range<1> gws (outer_size * NUM_THREADS);
  range<1> lws (NUM_THREADS);

  auto start = std::chrono::steady_clock::now();

  if (C <= 32) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm32>(
          nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          ChannelShuffleNHWCKernel<float, 32>(item, G, K, X, Y);
        });
      });
    }
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm128>(
          nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          ChannelShuffleNHWCKernel<float, 128>(item, G, K, X, Y);
        });
      });
    }
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm512>(
          nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          ChannelShuffleNHWCKernel<float, 512>(item, G, K, X, Y);
        });
      });
    }
  }

  q.wait();
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      size_t numel = N * C * W * H; 
      //size_t data_size_bytes = numel * sizeof(float);

      float *d_X = malloc_device<float>(numel, q);
      float *d_Y = malloc_device<float>(numel, q);
      if (d_X == nullptr || d_Y == nullptr) {
        if (d_X != nullptr) free(d_X, q);
        if (d_Y != nullptr) free(d_Y, q);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      auto ok = ChannelShuffleNHWC (q, d_X, N, C, G, numel, d_Y, time, repeat);
      if (ok)
        printf("Average time of channel shuffle (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
      else
        printf("Failed to execute channel shuffle (nhwc)\n");

      ok = ChannelShuffleNCHW (q, d_X, N, C, G, numel, d_Y, time, repeat);
      if (ok)
        printf("Average time of channel shuffle (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
      else
        printf("Failed to execute channel shuffle (nchw)\n");

      free(d_X, q);
      free(d_Y, q);
    }
  }
  
  end: return 0;
}
