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

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "common.h"
#include "reference.h"

template <typename T>
class test_atomics;

template <class T>
void testcase(queue &q, const int repeat)
{
  unsigned int len = 1 << 27;
  unsigned int localWorkSize = 256;
  unsigned int globalWorkSize = (len + localWorkSize - 1) /
                                localWorkSize * localWorkSize;
  unsigned int numData = 7;
  T gpuData[] = {0, 0, (T)-256, 256, 255, 0, 255};

  // allocate device memory for result
  buffer<T, 1> dOData(numData);

  range<1> gws (globalWorkSize);
  range<1> lws (localWorkSize);

  for (int i = 0; i < repeat; i++) {
    q.submit([&](handler &h) {
      auto d = dOData.template get_access<sycl_discard_write>(h);
      h.copy(gpuData, d);
    });

    q.submit([&](handler &h) {
      auto gpuData = dOData.template get_access<sycl_atomic>(h);
      h.parallel_for<class test_atomics<T>>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        int i = item.get_global_id(0);
        atomic_fetch_add(gpuData[0], (T)10);
        atomic_fetch_sub(gpuData[1], (T)10);
        atomic_fetch_max(gpuData[2], (T)i);
        atomic_fetch_min(gpuData[3], (T)i);
        atomic_fetch_and(gpuData[4], (T)(2*i+7));
        atomic_fetch_or (gpuData[5], (T)(1<<i));
        atomic_fetch_xor(gpuData[6], (T)i);
      });
    });
  }

  q.submit([&](handler &h) {
    auto d = dOData.template get_access<sycl_read>(h);
    h.copy(d, gpuData);
  }).wait();

  computeGold<T>(gpuData, globalWorkSize);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  const int repeat = atoi(argv[1]);
  testcase<int>(q, repeat);
  testcase<unsigned int>(q, repeat);
  return 0;
}
