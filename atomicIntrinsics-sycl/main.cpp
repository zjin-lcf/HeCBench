/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A simple program demonstrating trivial use of global memory atomic
 * device functions (atomic*() functions).
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

#define ATOMIC_REF(v) \
  sycl::atomic_ref<T, sycl::memory_order::relaxed, \
                   sycl::memory_scope::device,\
                   sycl::access::address_space::generic_space>(v)

template <class T>
void testKernel(sycl::nd_item<1> &item, T *g_odata)
{
  const int i = item.get_global_id(0);

  auto ao0 = ATOMIC_REF(g_odata[0]);
  ao0.fetch_add((T)10);

  auto ao1 = ATOMIC_REF(g_odata[1]);
  ao1.fetch_sub((T)10);

  auto ao2 = ATOMIC_REF(g_odata[2]);
  ao2.fetch_max((T)i);

  auto ao3 = ATOMIC_REF(g_odata[3]);
  ao3.fetch_min((T)i);

  auto ao4 = ATOMIC_REF(g_odata[4]);
  ao4.fetch_and((T)(2*i+7));

  auto ao5 = ATOMIC_REF(g_odata[5]);
  ao5.fetch_or((T)(1<<i));

  auto ao6 = ATOMIC_REF(g_odata[6]);
  ao6.fetch_xor((T)(i));

  // atomicInc and atomicDec are not fully supported across
  // vendors' GPUs. The implementations are from Syclomatic.
  /*
  auto ao7 = ATOMIC_REF(g_odata[7]);
  while (true) {
    T old = ao7.load();
    if (old >= 17) {
      if (ao7.compare_exchange_strong(old, 0))
        break;
    } else if (ao7.compare_exchange_strong(old, old + 1))
      break;
  }

  auto ao8 = ATOMIC_REF(g_odata[8]);
  while (true) {
    T old = ao8.load();
    if (old <= 0) {
      if (ao8.compare_exchange_strong(old, 137))
        break;
    } else if (ao8.compare_exchange_strong(old, old - 1))
      break;
  }
  */
}

template <typename T>
void testcase(sycl::queue &q, const int repeat)
{
  //unsigned int len = 1 << 27;
  unsigned int len = 1 << 16;
  unsigned int localWorkSize = 256;
  unsigned int globalWorkSize = (len + localWorkSize - 1) /
                                localWorkSize * localWorkSize;
  unsigned int numData = 9;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  unsigned int memSize = sizeof(gpuData);

  // allocate device memory for result
  T *dOData = sycl::malloc_device<T>(numData, q);

  sycl::range<1> gws (globalWorkSize);
  sycl::range<1> lws (localWorkSize);

  for (int i = 0; i < repeat; i++) {
    q.memcpy(dOData, gpuData, memSize);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        testKernel(item, dOData);
      });
    });
  }
  q.wait();

  //Copy result from device to host
  q.memcpy(gpuData, dOData, memSize).wait();

  computeGold<T>(gpuData, globalWorkSize);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        testKernel(item, dOData);
      });
    });
  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(dOData, q);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int repeat = atoi(argv[1]);
  testcase<int>(q, repeat);
  testcase<unsigned int>(q, repeat);
  return 0;
}
