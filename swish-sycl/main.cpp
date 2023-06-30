#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

#define GPU_THREADS 256

#define KERNEL_LOOP(index, range) \
  for (int index = item.get_global_id(0);  \
           index < (range); \
           index += item.get_local_range(0) * item.get_group_range(0))

template <typename T>
void SwishKernel(sycl::nd_item<1> &item, const int N, const T* X, T* Y)
{
  KERNEL_LOOP(i, N) {
    Y[i] = ldg(X + i) / (T(1) + sycl::exp(-ldg(X + i)));
  }
}

template <typename T>
void SwishGradientKernel(
    sycl::nd_item<1> &item,
    const int N,
    const T* X,
    const T* Y,
    const T* dY,
          T* dX)
{
  KERNEL_LOOP(i, N) {
    dX[i] = ldg(dY + i) *
            (ldg(Y + i) + (T(1) - ldg(Y + i)) / (T(1) + sycl::exp(-ldg(X + i))));
  }
}

template<typename T>
void eval_swish (const int N, const int repeat) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

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
  d_X = sycl::malloc_device<T>(N, q);
  q.memcpy(d_X, h_X, size_bytes);

  d_Y = sycl::malloc_device<T>(N, q);

  d_dY = sycl::malloc_device<T>(N, q);
  q.memcpy(d_dY, h_dY, size_bytes);

  d_dX = sycl::malloc_device<T>(N, q);

  sycl::range<1> gws ((N + GPU_THREADS - 1) / GPU_THREADS * GPU_THREADS);
  sycl::range<1> lws (GPU_THREADS);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class swish>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        SwishKernel(item, N, d_X, d_Y);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Swish kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class swish_grad>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        SwishGradientKernel(item, N, d_X, d_Y, d_dY, d_dX);
      });
    });
  };

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SwishGradient kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  q.memcpy(h_dX, d_dX, size_bytes);
  q.memcpy(h_Y, d_Y, size_bytes);
  q.wait();

  reference (N, h_X, r_Y, r_dX, h_dY);

  bool ok = true;
  for (int i = 0; i < N; i++) {
    if (fabs(h_dX[i] - r_dX[i]) > 1e-3 || fabs(h_Y[i] - r_Y[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_X, q);
  sycl::free(d_Y, q);
  sycl::free(d_dX, q);
  sycl::free(d_dY, q);

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
