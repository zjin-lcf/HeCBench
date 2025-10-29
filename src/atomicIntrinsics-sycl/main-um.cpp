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
#include "kernel.h"


template <typename T>
void testcase(sycl::queue &q, const int num, const int repeat)
{
  size_t len = 1UL << num;
  unsigned int localWorkSize = 256;
  unsigned int globalWorkSize = (len + localWorkSize - 1) /
                                localWorkSize * localWorkSize;
  unsigned int numData = 9;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255, 0, 0};
  unsigned int memSize = sizeof(gpuData);

  // allocate device memory for result
  T *dOData = sycl::malloc_shared<T>(numData, q);

  sycl::range<1> gws (globalWorkSize);
  sycl::range<1> lws (localWorkSize);

  for (int i = 0; i < repeat; i++) {
    q.memcpy(dOData, gpuData, memSize);

    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        testKernel(item, dOData, len);
      });
    });
  }
  q.wait();

  computeGold<T>(dOData, globalWorkSize);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        testKernel(item, dOData, len);
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
  if (argc != 3) {
    printf("Usage: %s <number of atomic operations> <repeat>\n", argv[0]);
    return 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int num = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  testcase<int>(q, num, repeat);
  testcase<unsigned int>(q, num, repeat);
  return 0;
}
