#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <assert.h>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n, sycl::nd_item<1> &item)
{
  unsigned int tid = item.get_global_id(0);
  for ( unsigned int i = tid * size; i < (tid + 1) * size; i++){
    result[tid] += i % 2;
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n, sycl::nd_item<1> &item)
{
  unsigned int tid = item.get_global_id(0);
  auto ao = sycl::atomic_ref<T, 
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (result[tid/size]);
  ao.fetch_add(tid%2); // arbitrary number to add
}

template <typename T>
class noAtomicKernel;

template <typename T>
class atomicKernel;

template <typename T>
void atomicCost (int t, int repeat)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  
  for (int size = 1; size <= 16; size++) {

    printf("\n\n");
    printf("Each thread sums up %d elements\n", size);

    assert(t % size == 0);
    assert(t / size % BLOCK_SIZE == 0);

    size_t result_size = sizeof(T) * t / size;

    T* result_wi = (T*) malloc (result_size);
    T* result_wo = (T*) malloc (result_size);

    T* d_result = (T *)sycl::malloc_device(result_size, q);
    
    sycl::range<1> lws (BLOCK_SIZE);
    sycl::range<1> gws_wo (t / size);
    sycl::range<1> gws_wi (t);

    q.wait();
    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      q.memset(d_result, 0, result_size);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class noAtomicKernel<T>>(
          sycl::nd_range<1>(gws_wi, lws), [=](sycl::nd_item<1> item) {
          wiAtomicOnGlobalMem<T>(d_result, size, t, item);
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    q.memcpy(result_wi, d_result, result_size).wait();

    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      q.memset(d_result, 0, result_size);
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class atomicKernel<T>>(
          sycl::nd_range<1>(gws_wo, lws), [=](sycl::nd_item<1> item) {
          woAtomicOnGlobalMem<T>(d_result, size, t, item);
        });
      });
    }
    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    q.memcpy(result_wo, d_result, result_size).wait();

    int diff = memcmp(result_wi, result_wo, result_size);
    printf("%s\n", diff ? "FAIL" : "PASS"); 

    free(result_wi);
    free(result_wo);
    free(d_result, q);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int t = 922521600;
  assert(t % BLOCK_SIZE == 0);
  
  printf("\nFP64 atomic add\n");
  atomicCost<double>(t, repeat); 

  printf("\nINT32 atomic add\n");
  atomicCost<int>(t, repeat); 

  printf("\nFP32 atomic add\n");
  atomicCost<float>(t, repeat); 

  return 0;
}
