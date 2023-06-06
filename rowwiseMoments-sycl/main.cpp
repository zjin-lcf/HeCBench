#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <type_traits>
#include <sycl/sycl.hpp>
#include "utils.h"

template <typename T>
void RowwiseMomentsKernel(
    int64_t N,
    T eps,
    const T* X,
    T* mean,
    T* rstd,
    const sycl::nd_item<1> &item)
{
  using T_ACC = T;
  using WelfordType = WelfordData<T_ACC, int64_t>;
  using WelfordOp = WelfordOps<T_ACC, T_ACC, int64_t, std::pair<T_ACC, T_ACC>>;

  sycl::multi_ptr<typename std::aligned_storage<
                  sizeof(WelfordType), alignof(WelfordType)>::type[WARP_SIZE],
                  sycl::access::address_space::local_space>
    localPtr = sycl::ext::oneapi::group_local_memory_for_overwrite<
        typename std::aligned_storage<sizeof(WelfordType), alignof(WelfordType)>::type[WARP_SIZE]>(item.get_group());
  WelfordType* val_shared_ptr = reinterpret_cast<WelfordType*>(*localPtr);

  const int64_t i = item.get_group(0);
  WelfordOp welford_op = {/*correction=*/0, /*take_sqrt=*/false};
  WelfordType val(0, 0, 0, 0);
  for (int64_t j = item.get_local_id(0); j < N; j += item.get_local_range(0)) {
    const int64_t index = i * N + j;
    val = welford_op.reduce(val, static_cast<T_ACC>(X[index]), index);
  }

  val = BlockReduce(
      val, welford_op,
      /*identity_element=*/WelfordType(0, 0, 0, 0),
      val_shared_ptr, item);

  if (item.get_local_id(0) == 0) {
    T_ACC m1;
    T_ACC m2;
    std::tie(m1, m2) = welford_op.project(val);
    rstd[i] = sycl::rsqrt(m1 + static_cast<T_ACC>(eps));
    mean[i] = m2;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 7) {
    printf("Usage: %s <batch> <channel> <width> <height> <group> <repeat>\n", argv[0]);
    return 1;
  }
  const int N = atoi(argv[1]);
  const int C = atoi(argv[2]);
  const int W = atoi(argv[3]);
  const int H = atoi(argv[4]);
  const int G = atoi(argv[5]);
  const int repeat = atoi(argv[6]);

  const int64_t D = C / G;

  double eps = 1e-6;

  size_t input_size = (size_t)N * C * W * H;
  size_t input_size_bytes = input_size * sizeof(float);

  size_t output_size = N * G;
  size_t output_size_bytes = output_size * sizeof(float);

  float* h_X = (float*) malloc (input_size_bytes);
  float* h_mean = (float*) malloc (output_size_bytes);
  float* h_rstd = (float*) malloc (output_size_bytes);

  srand(123);
  for (size_t i = 0; i < input_size; i++) {
    h_X[i] = rand() / (float)RAND_MAX;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_X;
  d_X = (float *)sycl::malloc_device(input_size_bytes, q);
  q.memcpy(d_X, h_X, input_size_bytes);

  float* d_mean, *d_rstd;
  d_mean = (float *)sycl::malloc_device(output_size_bytes, q);
  d_rstd = (float *)sycl::malloc_device(output_size_bytes, q);

  sycl::range<1> gws (N * G * kNumThreads);
  sycl::range<1> lws (kNumThreads);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(32)]] {
        RowwiseMomentsKernel<float>(D * H * W, eps, d_X, d_mean, d_rstd, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of RowwiseMoments kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  q.memcpy(h_mean, d_mean, output_size_bytes);
  q.memcpy(h_rstd, d_rstd, output_size_bytes);
  q.wait();

  double avg_rstd = 0.0, avg_mean = 0.0;
  for (size_t i = 0; i < output_size; i++) {
    avg_mean += h_mean[i];
    avg_rstd += h_rstd[i];
  }
  avg_rstd /= output_size;
  avg_mean /= output_size;

  printf("Checksum: mean = %lf and rstd = %lf\n", avg_mean, avg_rstd);

  sycl::free(d_X, q);
  sycl::free(d_mean, q);
  sycl::free(d_rstd, q);

  free(h_X);
  free(h_mean);
  free(h_rstd);
  return 0;
}
