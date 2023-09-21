#include <chrono>
#include <cstdio>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include "kernels.h"

using namespace std::chrono;

void Forward(sycl::queue &q, int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 6;
  int64_t C = 150;
  int64_t D = 100;
  int64_t H = 160;
  int64_t W = 160;
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) {
    src[i] = rand() / (float)RAND_MAX;
  }

  size_t bytes_to_copy_s = wk_size * sizeof(float);
  float *src_mem = sycl::malloc_device<float>(wk_size, q);
  q.memcpy(src_mem, src.data(), bytes_to_copy_s);

  size_t bytes_to_copy_d = wk_size * sizeof(float);
  float *dst_mem = sycl::malloc_device<float>(wk_size, q);
  q.memcpy(dst_mem, dst.data(), bytes_to_copy_d);

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    sycl::range<1> gws (wg_size * wg_cnt);
    sycl::range<1> lws (wg_size);

    q.wait();
    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class fwd>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          lrn_fwd_kernel(item, src_mem, dst_mem, N, C, D, H, W,
                         stride_mb, ndims, wk_size, size, alpha, beta, k);
        });
      });
    }

    q.wait();
    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_fwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (2 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }

  q.memcpy(dst.data(), dst_mem, bytes_to_copy_d).wait();
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) {
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);

  free(src_mem, q);
  free(dst_mem, q);
}

void Backward(sycl::queue &q, int repeat)
{
  int64_t ndims = 5;
  int64_t size = 5;
  float alpha = 0.000122;
  float beta = 0.750000;
  float k = 1.000000;
  int64_t N = 5;
  int64_t C = 150;
  int64_t D = 100;
  int64_t H = 160;
  int64_t W = 160;
  int64_t stride_mb = C*D*H*W;
  int64_t wk_size = N*C*D*H*W;

  std::vector<float> src(wk_size, 0);
  std::vector<float> dst(wk_size, 0);
  std::vector<float> diff_src(wk_size, 0);

  srand(123);
  for (int64_t i = 0; i < wk_size; i++) {
    diff_src[i] = src[i] = rand() / (float)RAND_MAX;
  }

  size_t bytes_to_copy_s = wk_size * sizeof(float);
  float *src_mem = sycl::malloc_device<float>(wk_size, q);
  q.memcpy(src_mem, src.data(), bytes_to_copy_s);

  size_t bytes_to_copy_diff = wk_size * sizeof(float);
  float *diff_src_mem = sycl::malloc_device<float>(wk_size, q);
  q.memcpy(diff_src_mem, diff_src.data(), bytes_to_copy_diff);

  size_t bytes_to_copy_d = wk_size * sizeof(float);
  float *dst_mem = sycl::malloc_device<float>(wk_size, q);
  q.memcpy(dst_mem, src.data(), bytes_to_copy_d);

  printf("Sweep the work-group sizes from 64 to 512\n");
  for (int wg_size = 64; wg_size <= 512; wg_size = wg_size * 2) {

    int64_t wg_cnt = (wk_size + wg_size - 1) / wg_size;

    sycl::range<1> gws (wg_size * wg_cnt);
    sycl::range<1> lws (wg_size);

    q.wait();
    auto start = high_resolution_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class bwd>(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
           lrn_bwd_kernel(item, src_mem, dst_mem, diff_src_mem, N, C,
                          D, H, W, stride_mb, ndims, wk_size,
                          size, alpha, beta, k);
         });
      });
    }

    q.wait();
    auto stop = high_resolution_clock::now();

    auto time = (duration_cast<microseconds>(stop - start)).count()/1e6f;
    printf("Average execution time of lrn_bwd_kernel: %.6f sec \n", time / repeat);

    auto data_inGB = (3 * wk_size * sizeof(float)) / 1e9f;
    auto bandwidth = data_inGB * repeat / time;

    printf("Kernel bandwidth: %.6f GB/s \n", bandwidth);
  }

  q.memcpy(dst.data(), dst_mem, bytes_to_copy_d).wait();
  double checksum = 0;
  for (int64_t i = 0; i < wk_size; i++) {
    checksum += dst[i];
  }
  printf("Checksum: %lf\n", checksum / wk_size);
  free(src_mem, q);
  free(diff_src_mem, q);
  free(dst_mem, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Forward(q, repeat);
  Backward(q, repeat);

  return 0;
}
