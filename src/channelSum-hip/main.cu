#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

template <typename T>
using BlockReduce1D = hipcub::BlockReduce<T, NUM_THREADS>;

template <typename T, int kBlockDimX, int kBlockDimY>
using BlockReduce2D = hipcub::BlockReduce<T,
                                       kBlockDimX,
                                       hipcub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                       kBlockDimY>;

#define DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(                      \
    size, Func, T, grid_dim, stream, ...)                                  \
  do {                                                                     \
    if (size >= 128) {                                                     \
      Func<T, 1, 128><<<grid_dim, dim3(1, 128), 0, stream>>>(__VA_ARGS__); \
    } else if (size >= 64) {                                               \
      Func<T, 2, 64><<<grid_dim, dim3(2, 64), 0, stream>>>(__VA_ARGS__);   \
    } else if (size >= 32) {                                               \
      Func<T, 4, 32><<<grid_dim, dim3(4, 32), 0, stream>>>(__VA_ARGS__);   \
    } else {                                                               \
      Func<T, 8, 16><<<grid_dim, dim3(8, 16), 0, stream>>>(__VA_ARGS__);   \
    }                                                                      \
  } while (false)


template <typename T, int kBlockDimX, int kBlockDimY>
__global__
void ChannelSumNCHW(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  __shared__
  typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage m_storage;

  __shared__
  typename BlockReduce2D<T, kBlockDimX, kBlockDimY>::TempStorage v_storage;

  T m_val = 0;
  T v_val = 0;

  const int c = blockIdx.x;

  // sum batches from different channels
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    for (int hw = threadIdx.y; hw < HxW; hw += blockDim.y) {
      const int index = (n * C + c) * HxW + hw;
      m_val += __ldg(X + index);
      v_val += __ldg(X + index) * __ldg(X + index);
    }
  }
  m_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(m_storage).Sum(m_val);
  v_val = BlockReduce2D<T, kBlockDimX, kBlockDimY>(v_storage).Sum(v_val);

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

template <typename T>
__global__
void ChannelSumNHWC(
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  __shared__ typename BlockReduce1D<T>::TempStorage m_storage;
  __shared__ typename BlockReduce1D<T>::TempStorage v_storage;
  const int inner_size = N * HxW;
  const int c = blockIdx.x;
  T m_val = 0;
  T v_val = 0;
  for (int i = threadIdx.x; i < inner_size; i += blockDim.x) {
    const int index = i * C + c;
    m_val += __ldg(X + index);
    v_val += __ldg(X + index) * __ldg(X + index);
  }
  m_val = BlockReduce1D<T>(m_storage).Sum(m_val);
  v_val = BlockReduce1D<T>(v_storage).Sum(v_val);
  if (threadIdx.x == 0) {
    sum[c] = m_val;
    sumsq[c] = v_val;
  }
}

void ComputeChannelSumNCHW (
    const int N,
    const int C,
    const int HxW,
    const scalar_t* X,
    scalar_t* sum,
    scalar_t* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(
      HxW,
      ChannelSumNCHW,
      scalar_t,
      C,
      0,
      N,
      C,
      HxW,
      X,
      sum,
      sumsq);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ComputeChannelSumNHWC (
    const int N,
    const int C,
    const int HxW,
    const scalar_t* X,
    scalar_t* sum,
    scalar_t* sumsq,
    long &time,
    int repeat)
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNHWC<scalar_t> <<<C, NUM_THREADS>>>(N, C, HxW, X, sum, sumsq);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int W = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  long time;

  for (int N = 1; N <= 64; N = N * 4) {
    for (int C = 32; C <= 512; C = C * 4) {

      printf("\n(N=%d C=%d W=%d H=%d)\n", N, C, W, H);

      int numel = N * C * W * H; // assume no integer overflow

      size_t data_size_bytes = numel * sizeof(scalar_t);
      size_t output_size_bytes = C * sizeof(scalar_t);

      scalar_t *h_X = (scalar_t*) malloc (data_size_bytes);
      scalar_t *h_sum = (scalar_t*) malloc (output_size_bytes);
      scalar_t *h_sumsq = (scalar_t*) malloc (output_size_bytes);
      scalar_t *r_sum = (scalar_t*) malloc (output_size_bytes);

      srand(numel);
      for (int i = 0; i < numel; i++) h_X[i] = rand() % 256;

      scalar_t *d_X, *d_sum, *d_sumsq;
      auto errorX = hipMalloc((void**)&d_X, data_size_bytes);
      auto errorY = hipMalloc((void**)&d_sum, output_size_bytes);
      auto errorZ = hipMalloc((void**)&d_sumsq, output_size_bytes);
      if (errorX != hipSuccess ||
          errorY != hipSuccess ||
          errorZ != hipSuccess) {
        if (errorX == hipSuccess) hipFree(d_X);
        if (errorY == hipSuccess) hipFree(d_sum);
        if (errorZ == hipSuccess) hipFree(d_sumsq);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      hipMemcpy(d_X, h_X, data_size_bytes, hipMemcpyHostToDevice);

      ComputeChannelSumNHWC (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      hipMemcpy(h_sum, d_sum, output_size_bytes, hipMemcpyDeviceToHost);
      ref_nhwc (N, C, W*H, h_X, r_sum, h_sumsq);
      bool ok = check(C, h_sum, r_sum);

      if (ok)
        printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
      else {
        printf("Verification fails for channel sum (nhwc)\n");
        exit(1);
      }

      ComputeChannelSumNCHW (N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      hipMemcpy(h_sum, d_sum, output_size_bytes, hipMemcpyDeviceToHost);
      ref_nchw (N, C, W*H, h_X, r_sum, h_sumsq);
      ok = check(C, h_sum, r_sum);
      
      if (ok)
        printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
      else {
        printf("Verification fails for channel sum (nchw)\n");
        exit(1);
      }

      hipFree(d_X);
      hipFree(d_sum);
      hipFree(d_sumsq);

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  
  end: return 0;
}
