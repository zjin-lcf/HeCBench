#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <functional>
#include <sycl/sycl.hpp>
#include "reference.h"

#define NUM_THREADS 256

// choose integer type to avoid floating-point rounding errors
typedef int scalar_t;

#define DISPATCH_REDUCE_KERNEL_BY_2D_BLOCK_WITH_TYPE(                  \
   size, Func, T, grid_dim, ...)                                       \
  do {                                                                 \
   if (size >= 128) {                                                  \
     sycl::range<3> gws (1, 128, grid_dim);                            \
     sycl::range<3> lws (1, 128, 1);                                   \
     Func<T>(q, gws, lws, 0, __VA_ARGS__);                             \
   } else if (size >= 64) {                                            \
     sycl::range<3> gws (1, 64, 2 * grid_dim);                         \
     sycl::range<3> lws (1, 64, 2);                                    \
     Func<T>(q, gws, lws, 0, __VA_ARGS__);                             \
   } else if (size >= 32) {                                            \
     sycl::range<3> gws (1, 32, 4 * grid_dim);                         \
     sycl::range<3> lws (1, 32, 4);                                    \
     Func<T>(q, gws, lws, 0, __VA_ARGS__);                             \
   } else {                                                            \
     sycl::range<3> gws (1, 16, 8 * grid_dim);                         \
     sycl::range<3> lws (1, 16, 8);                                    \
     Func<T>(q, gws, lws, 0, __VA_ARGS__);                             \
   }                                                                   \
  } while (false)


template <typename T>
void BlockReduce(T &input1, T &input2, sycl::nd_item<3> &item) {
  input1 = sycl::reduce_over_group(item.get_group(), input1, sycl::plus<T>());
  input2 = sycl::reduce_over_group(item.get_group(), input2, sycl::plus<T>());
}

template <typename T>
void ChannelSumNCHW(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      T m_val = 0;
      T v_val = 0;

      const int c = item.get_group(2);

      // sum batches from different channels
      for (int n = item.get_local_id(2); n < N;
               n += item.get_local_range(2)) {
        for (int hw = item.get_local_id(1); hw < HxW;
             hw += item.get_local_range(1)) {
          const int index = (n * C + c) * HxW + hw;
          m_val += *(X + index);
          v_val += *(X + index) * *(X + index);
        }
      }

      BlockReduce<T>(m_val, v_val, item);
      if (item.get_local_id(2) == 0 && item.get_local_id(1) == 0) {
        sum[c] = m_val;
        sumsq[c] = v_val;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void ChannelSumNHWC(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    const int N,
    const int C,
    const int HxW,
    const T* X,
    T*__restrict__ sum,
    T*__restrict__ sumsq)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      const int inner_size = N * HxW;
      const int c = item.get_group(2);
      T m_val = 0;
      T v_val = 0;
      for (int i = item.get_local_id(2); i < inner_size;
           i += item.get_local_range(2)) {
        const int index = i * C + c;
        m_val += *(X + index);
        v_val += *(X + index) * *(X + index);
      }

      BlockReduce<T>(m_val, v_val, item);
      if (item.get_local_id(2) == 0) {
        sum[c] = m_val;
        sumsq[c] = v_val;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void ComputeChannelSumNCHW (
    sycl::queue &q,
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
        HxW, ChannelSumNCHW, scalar_t, C, N, C, HxW, X, sum, sumsq);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

void ComputeChannelSumNHWC (
    sycl::queue &q,
    const int N,
    const int C,
    const int HxW,
    const scalar_t* X,
    scalar_t* sum,
    scalar_t* sumsq,
    long &time,
    int repeat)
{
  sycl::range<3> lws (1, 1, NUM_THREADS);
  sycl::range<3> gws (1, 1, C * NUM_THREADS);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    ChannelSumNHWC<scalar_t>(q, gws, lws, 0, N, C, HxW, X, sum, sumsq);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int W = atoi(argv[1]);
  const int H = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  long time;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

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
      d_X = sycl::malloc_device<scalar_t>(numel, q);
      d_sum = sycl::malloc_device<scalar_t>(C, q);
      d_sumsq = sycl::malloc_device<scalar_t>(C, q);
      if (d_X == nullptr || d_sum == nullptr || d_sumsq == nullptr) {
        if (d_X != nullptr) sycl::free(d_X, q);
        if (d_sum != nullptr) sycl::free(d_sum, q);
        if (d_sumsq != nullptr) sycl::free(d_sumsq, q);
        printf("Device memory allocation failed. Exit\n");
        goto end;
      }

      q.memcpy(d_X, h_X, data_size_bytes).wait();

      ComputeChannelSumNHWC (q, N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      q.memcpy(h_sum, d_sum, output_size_bytes).wait();
      ref_nhwc (N, C, W*H, h_X, r_sum, h_sumsq);
      bool ok = check(C, h_sum, r_sum);

      printf("Average time of channel sum (nhwc): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nhwc)\n", ok ? "PASS" : "FAIL");

      ComputeChannelSumNCHW (q, N, C, W*H, d_X, d_sum, d_sumsq, time, repeat);

      q.memcpy(h_sum, d_sum, output_size_bytes).wait();
      ref_nchw (N, C, W*H, h_X, r_sum, h_sumsq);
      ok = check(C, h_sum, r_sum);
      
      printf("Average time of channel sum (nchw): %f (ms)\n", (time * 1e-6f) / repeat);
      printf("Verification %s for channel sum (nchw)\n", ok ? "PASS" : "FAIL");

      sycl::free(d_X, q);
      sycl::free(d_sum, q);
      sycl::free(d_sumsq, q);

      free(h_X);
      free(h_sum);
      free(r_sum);
      free(h_sumsq);
    }
  }
  
  end: return 0;
}
