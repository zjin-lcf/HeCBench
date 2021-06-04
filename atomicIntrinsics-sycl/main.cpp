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



#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)


////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold(int *gpuData, const int len)
{
    int val = 0;

    for (int i = 0; i < len; ++i)
    {
        val += 10;
    }

    if (val != gpuData[0])
    {
        printf("Add failed %d %d\n", val, gpuData[0]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val -= 10;
    }

    if (val != gpuData[1])
    {
        printf("Sub failed: %d %d\n", val, gpuData[1]);
    }

    val = -(1<<8);

    for (int i = 0; i < len; ++i)
    {
        // fourth element should be len-1
        val = max(val, i);
    }

    if (val != gpuData[2])
    {
        printf("Max failed: %d %d\n", val, gpuData[2]);
    }

    val = 1 << 8;

    for (int i = 0; i < len; ++i)
    {
        val = min(val, i);
    }

    if (val != gpuData[3])
    {
        printf("Min failed: %d %d\n", val, gpuData[3]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val &= (2 * i + 7);
    }

    if (val != gpuData[4])
    {
        printf("And failed: %d %d\n", val, gpuData[4]);
    }

    val = 0;

    for (int i = 0; i < len; ++i)
    {
        val |= (1 << i);
    }

    if (val != gpuData[5])
    {
        printf("Or failed: %d %d\n", val, gpuData[5]);
    }

    val = 0xff;

    for (int i = 0; i < len; ++i)
    {
        val ^= i;
    }

    if (val != gpuData[6])
    {
        printf("Xor failed %d %d\n", val, gpuData[6]);
    }

    printf("PASS\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  unsigned int len = 1 << 27;
  unsigned int numThreads = 256;
  unsigned int numBlocks = (len + numThreads - 1) / numThreads * numThreads;
  unsigned int numData = 7;
  int gpuData[] = {0, 0, -(1<<8), 1<<8, 0xff, 0, 0xff};

#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  // allocate device memory for result
  buffer<int, 1> dOData(numData);

  range<1> global_work_size (numBlocks);
  range<1> local_work_size (numThreads);

  for (int i = 0; i < 1; i++) {
    q.submit([&](handler &h) {
      auto d = dOData.get_access<sycl_discard_write>(h);
      h.copy(gpuData, d);
    });
    q.submit([&](handler &h) {
      auto gpuData = dOData.get_access<sycl_atomic>(h);
      h.parallel_for<class test_atomics>(nd_range<1>(global_work_size, local_work_size), [=](nd_item<1> item) {
        int i = item.get_global_id(0);
        atomic_fetch_add(gpuData[0], 10);
        atomic_fetch_sub(gpuData[1], 10);
        atomic_fetch_max(gpuData[2], i);
        atomic_fetch_min(gpuData[3], i);
        atomic_fetch_and(gpuData[4], 2*i+7);
        atomic_fetch_or (gpuData[5], 1<<i);
        atomic_fetch_xor(gpuData[6], i);
        });
      });
  }
  q.submit([&](handler &h) {
    auto d = dOData.get_access<sycl_read>(h);
    h.copy(d, gpuData);
  });
  q.wait();

  computeGold(gpuData, numBlocks);
  return 0;
}
