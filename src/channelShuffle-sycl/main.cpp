#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

#define NUM_THREADS 256
#define GridDimMaxY 65536

template <typename T, bool kNFirst>
void ChannelShuffleNCHWKernel(
    sycl::nd_item<3> &item,
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
    sycl::nd_item<1> &item,
    const int G,
    const int K,
    const T* X,
          T* Y)
{
  int blockIdx_x  = item.get_group(0);
  int blockDim_x  = item.get_local_range(0);
  int threadIdx_x = item.get_local_id(0);

  auto g = item.get_group();
  sycl::multi_ptr<T[kSharedSize], sycl::access::address_space::local_space> localPtr =
    sycl::ext::oneapi::group_local_memory_for_overwrite<T[kSharedSize]>(g);

  T* sdata = *localPtr;

  const int C = G * K;
  const int offset = blockIdx_x * C;
  for (int i = threadIdx_x; i < C; i += blockDim_x) {
    sdata[i] = X[offset + i];
  }

  sycl::group_barrier(g, sycl::memory_scope::work_group);

  for (int i = threadIdx_x; i < C; i += blockDim_x) {
    const int g = i % G;
    const int k = i / G;
    Y[offset + i] = sdata[g * K + k];
  }
}

template <typename T>
bool ChannelShuffleNCHW (sycl::queue &q, T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int S = (HxW + NUM_THREADS - 1) / NUM_THREADS;

  auto start = std::chrono::steady_clock::now();

  if (N <= GridDimMaxY) {
    sycl::range<3> gws (C, N, S * NUM_THREADS);
    sycl::range<3> lws (1, 1, NUM_THREADS);

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class shuffle_nchw>(
          sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
          ChannelShuffleNCHWKernel<float, false>(item, G, K, HxW, X, Y);
        });
      });
    }
  } else {
    sycl::range<3> gws (C, S, N * NUM_THREADS);
    sycl::range<3> lws (1, 1, NUM_THREADS);

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class shuffle2_nchw>(
          sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
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
bool ChannelShuffleNHWC (sycl::queue &q, T *X, int N, int C, int G, int numel, T *Y,
                         long &time, int repeat)
{
  if (C % G != 0 || numel < N * C) return false;

  const int K = C / G;
  const int HxW = numel / (N * C);
  const int outer_size = N * HxW;

  sycl::range<1> gws (outer_size * NUM_THREADS);
  sycl::range<1> lws (NUM_THREADS);

  auto start = std::chrono::steady_clock::now();

  if (C <= 32) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm32>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          ChannelShuffleNHWCKernel<float, 32>(item, G, K, X, Y);
        });
      });
    }
  } else if (C <= 128) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm128>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          ChannelShuffleNHWCKernel<float, 128>(item, G, K, X, Y);
        });
      });
    }
  } else if (C <= 512) {
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class shuffle_nhwc_sm512>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
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
  float *h_X, *h_Y, *h_Y_ref;
  float *d_X, *d_Y;
  int error;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // limited by the global device memory
  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);
      size_t data_size_bytes = sizeof(float) * N * C * W * H;

      const int numel = N * C * W * H; // assume no integer overflow

      d_X = sycl::malloc_device<float>(numel, q);
      d_Y = sycl::malloc_device<float>(numel, q);
      if (d_X == nullptr || d_Y == nullptr) {
        if (d_X != nullptr) sycl::free(d_X, q);
        if (d_Y != nullptr) sycl::free(d_Y, q);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      h_X = (float*) malloc(data_size_bytes);
      for (int i = 0; i < numel; i++) h_X[i] = (float) i / numel;

      h_Y = (float*) malloc(data_size_bytes);
      h_Y_ref = (float*) malloc(data_size_bytes);

      q.memcpy(d_X, h_X, data_size_bytes);

      ChannelShuffleNHWC (q, d_X, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNHWC_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      q.memcpy(h_Y, d_Y, data_size_bytes).wait();
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NHWC) check\n");
      else
        printf("Average time of channel shuffle (NHWC): %f (ms)\n", (time * 1e-6f) / repeat);

      ChannelShuffleNCHW (q, d_X, N, C, G, numel, d_Y, time, repeat);
      ChannelShuffleNCHW_cpu (h_X, N, C, G, numel, h_Y_ref, time, repeat);
      q.memcpy(h_Y, d_Y, data_size_bytes).wait();
      error = memcmp(h_Y, h_Y_ref, data_size_bytes);
      if (error)
        printf("Failed to pass channel shuffle (NCHW) check\n");
      else
        printf("Average time of channel shuffle (NCHW): %f (ms)\n", (time * 1e-6f) / repeat);

      sycl::free(d_X, q);
      sycl::free(d_Y, q);
      free(h_X);
      free(h_Y);
      free(h_Y_ref);
    }
  }

  end: return 0;
}
