#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "atomics.h"

#define BLOCK_SIZE 256

#include "reference.h"

template <typename T>
void BlockRangeAtomicOnGlobalMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
        atomicAdd(data[item.get_local_id(2)], (T)1);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void WarpRangeAtomicOnGlobalMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
        atomicAdd(data[i & 0x1F], (T)1);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void SingleRangeAtomicOnGlobalMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int offset, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
            atomicAdd(data[offset], (T)1);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void BlockRangeAtomicOnSharedMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> smem_data (sycl::range<1>(BLOCK_SIZE), cgh);

    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
        atomicAdd(smem_data[item.get_local_id(2)], (T)1);
      }
      if (item.get_group(2) == item.get_group_range(2))
        data[item.get_local_id(2)] = smem_data[item.get_local_id(2)];
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void WarpRangeAtomicOnSharedMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> smem_data (sycl::range<1>(32), cgh);
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
        atomicAdd(smem_data[i & 0x1F], (T)1);
      }
      if (item.get_group(2) == item.get_group_range(2) &&
          item.get_local_id(2) < 0x1F)
        data[item.get_local_id(2)] = smem_data[item.get_local_id(2)];
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void SingleRangeAtomicOnSharedMem(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    T* data, int offset, int n)
{
  auto cgf = [&] (sycl::handler &cgh) {
    sycl::local_accessor<T, 1> smem_data (sycl::range<1>(BLOCK_SIZE), cgh);
    auto kfn = [=] (sycl::nd_item<3> item) {
      unsigned int tid = item.get_global_id(2);
      for (unsigned int i = tid; i < n;
           i += item.get_local_range(2) * item.get_group_range(2)) {
        atomicAdd(smem_data[offset], (T)1);
      }
      if (item.get_group(2) == item.get_group_range(2) &&
          item.get_local_id(2) == 0)
        data[item.get_local_id(2)] = smem_data[item.get_local_id(2)];
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);
  T* h_data = (T*) malloc (data_size);
  T* r_data = (T*) malloc (data_size);
  int fail;

  for(int i=0; i<t; i++) {
    data[i] = i%1024+1;
  }

  T* d_data = (T *)sycl::malloc_device(data_size, q);

  sycl::range<3> lws (1, 1, BLOCK_SIZE);
  sycl::range<3> gws (1, 1, n);

  q.memcpy(d_data, data, data_size).wait();
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnGlobalMem<T>(q, gws, lws, 0, d_data, n);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnGlobalMem<T>(q, gws, lws, 0, d_data, n);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnGlobalMem<T>(q, gws, lws, 0, d_data, i % BLOCK_SIZE, n);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i % BLOCK_SIZE, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnSharedMem<T>(q, gws, lws, 0, d_data, n);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnSharedMem<T>(q, gws, lws, 0, d_data, n);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  q.memcpy(d_data, data, data_size).wait();
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnSharedMem<T>(q, gws, lws, 0, d_data, i % BLOCK_SIZE, n);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(h_data, d_data, data_size).wait();
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
  sycl::free(d_data, q);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; // number of threads
  const int len = 1024; // data array length

  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat);

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat);

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat);

  return 0;
}
