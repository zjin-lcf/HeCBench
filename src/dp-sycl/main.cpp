/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between two arrays.
//
// Runs computations with on the GPU device and then checks results
// against basic host CPU/C++ computation.
// *********************************************************************

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/numeric>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>
#include "shrUtils.h"

typedef double Type;

// Forward Declarations
Type DotProductHost(const Type* pfData1, const Type* pfData2, size_t iNumElements);

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
  // rounded up to the nearest multiple of the LocalWorkSize
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(Type);

  const size_t grid_size = shrRoundUp(szLocalWorkSize,
                                      szGlobalWorkSize / (szLocalWorkSize * 4));

  // Allocate and initialize host arrays
  Type* srcA = (Type*) malloc (src_size_bytes);
  Type* srcB = (Type*) malloc (src_size_bytes);
  Type  dst;

  size_t i;
  srand(123);
  for (i = 0; i < iNumElements ; ++i)
  {
    srcA[i] = 1;
    srcB[i] = -1;
  }
  for (i = iNumElements; i < src_size ; ++i) {
    srcA[i] = srcB[i] = 0;
  }

  // Compute and compare results for golden-host and report errors and pass/fail
  Type Golden = DotProductHost (srcA, srcB, iNumElements);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  Type *d_srcA = sycl::malloc_device<Type>(src_size, q);
  q.memcpy(d_srcA, srcA, src_size_bytes);

  Type *d_srcB = sycl::malloc_device<Type>(src_size, q);
  q.memcpy(d_srcB, srcB, src_size_bytes);

  Type *d_dst = sycl::malloc_device<Type>(1, q);

  sycl::range<1> gws (grid_size * szLocalWorkSize);
  sycl::range<1> lws (szLocalWorkSize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    q.memset(d_dst, 0, sizeof(Type));
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class dot_product>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        size_t iGID = item.get_global_id(0);
        Type sum = 0;
        for(size_t idx = iGID; idx < src_size / 4;
            idx += item.get_local_range(0) * item.get_group_range(0)) {
          size_t iInOffset = idx * 4;
          sum += d_srcA[iInOffset    ] * d_srcB[iInOffset    ] +
                 d_srcA[iInOffset + 1] * d_srcB[iInOffset + 1] +
                 d_srcA[iInOffset + 2] * d_srcB[iInOffset + 2] +
                 d_srcA[iInOffset + 3] * d_srcB[iInOffset + 3];
        }
        Type aggregate = sycl::reduce_over_group(item.get_group(), sum, std::plus<>());
        if (item.get_local_id(0) == 0) {
           sycl::atomic_ref<Type, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> ao (d_dst[0]);
           ao.fetch_add(aggregate);
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iNumIterations);

  q.memcpy(&dst, d_dst, sizeof(Type)).wait();
  bool bMatch = std::abs(Golden - dst) < 1e-3f;
  printf("GPU Result %s CPU Result\n\n", bMatch ? "matches" : "DOESN'T match");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    dst = std::transform_reduce(oneapi::dpl::execution::make_device_policy(q),
                                d_srcA, d_srcA + iNumElements, d_srcB, .0);
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average std::transform_reduce execution time %f (s)\n", (time * 1e-9f) / iNumIterations);
  bMatch = std::abs(Golden - dst) < 1e-3f;
  printf("\nGPU Result %s CPU Result\n", bMatch ? "matches" : "DOESN'T match");

  sycl::free(d_dst, q);
  sycl::free(d_srcA, q);
  sycl::free(d_srcB, q);

  free(srcA);
  free(srcB);
  return EXIT_SUCCESS;
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
Type DotProductHost(const Type* pfData1, const Type* pfData2, size_t iNumElements)
{
  Type r = 0;
  for (size_t i = 0; i < iNumElements; i++)
  {
    r += pfData1[i] * pfData2[i];
  }
  return r;
}
