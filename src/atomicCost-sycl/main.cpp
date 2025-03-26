#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "atomics.h"

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* result, int size)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for ( unsigned int i = tid * size; i < (tid + 1) * size; i++){
        result[tid] += i % 2;
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* result, int size)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for ( unsigned int i = tid * size; i < (tid + 1) * size; i++){
        atomicAdd(result[tid], i % 2);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void atomicCost (int length, int size, int repeat)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  size_t result_size = sizeof(T) * num_threads;

  T* result_wi = (T*) malloc (result_size);
  T* result_wo = (T*) malloc (result_size);

  T* d_result_wi = (T *)sycl::malloc_device(result_size, q);
  q.memset(d_result_wi, 0, result_size);

  T* d_result_wo = (T *)sycl::malloc_device(result_size, q);
  q.memset(d_result_wo, 0, result_size);

  sycl::range<3> lws (1, 1, BLOCK_SIZE);
  sycl::range<3> gws (1, 1, num_threads);

  q.wait();
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    wiAtomicOnGlobalMem<T>(q, gws, lws, 0, d_result_wi, size);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);
  q.memcpy(result_wi, d_result_wi, result_size).wait();

  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    woAtomicOnGlobalMem<T>(q, gws, lws, 0, d_result_wo, size);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);
  q.memcpy(result_wo, d_result_wo, result_size).wait();

  int diff = memcmp(result_wi, result_wo, result_size);
  printf("%s\n", diff ? "FAIL" : "PASS");

  free(result_wi);
  free(result_wo);
  free(d_result_wi, q);
  free(d_result_wo, q);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}
