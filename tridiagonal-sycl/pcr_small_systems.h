/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Tridiagonal solvers.
 * Host code for parallel cyclic reduction (PCR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _PCR_SMALL_SYSTEMS_
#define _PCR_SMALL_SYSTEMS_

#include <sycl/sycl.hpp>
#include "tridiagonal.h"
#include "pcr_kernels.cpp"

const char *pcrKernelNames[] = {
  "pcr_small_systems_kernel",    // original version
  "pcr_branch_free_kernel",      // optimized branch-free version
};

double pcr_small_systems(sycl::queue &q, float *a, float *b, float *c, float *d, float *x,
    int system_size, int num_systems, int id = 0)
{
  shrLog(" %s\n", pcrKernelNames[id]);

  const unsigned int mem_size = num_systems * system_size;
  float *a_d = sycl::malloc_device<float>(mem_size, q);
  q.memcpy(a_d, a, mem_size * sizeof(float));

  float *b_d = sycl::malloc_device<float>(mem_size, q);
  q.memcpy(b_d, b, mem_size * sizeof(float));

  float *c_d = sycl::malloc_device<float>(mem_size, q);
  q.memcpy(c_d, c, mem_size * sizeof(float));

  float *d_d = sycl::malloc_device<float>(mem_size, q);
  q.memcpy(d_d, d, mem_size * sizeof(float));

  float *x_d = sycl::malloc_device<float>(mem_size, q);
  q.memcpy(x_d, x, mem_size * sizeof(float));

  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;
  int iterations = log2(system_size/2);

  // set execution parameters
  szLocalWorkSize = system_size;
  szGlobalWorkSize = num_systems * szLocalWorkSize;

  sycl::range<1> gws (szGlobalWorkSize);
  sycl::range<1> lws (szLocalWorkSize);

  q.wait();
  shrLog("  looping %i times..\n", BENCH_ITERATIONS);

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (id == 0)
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>((system_size+1)*5), cgh);
        cgh.parallel_for<class pcr>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          pcr_small_systems_kernel(item,
                                   a_d,
                                   b_d,
                                   c_d,
                                   d_d,
                                   x_d,
                                   lmem.get_pointer(),
                                   system_size,
                                   num_systems,
                                   iterations);
        });
      });
    else
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> lmem(sycl::range<1>((system_size+1)*5), cgh);
        cgh.parallel_for<class pcr_opt>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          pcr_branch_free_kernel(item,
                                 a_d,
                                 b_d,
                                 c_d,
                                 d_d,
                                 x_d,
                                 lmem.get_pointer(),
                                 system_size,
                                 num_systems,
                                 iterations);
        });
      });
  }
  q.wait();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  // write-back to the array x
  q.memcpy(x, x_d, mem_size * sizeof(float)).wait();

  sycl::free(a_d, q);
  sycl::free(b_d, q);
  sycl::free(c_d, q);
  sycl::free(d_d, q);
  sycl::free(x_d, q);
  return time;
}
#endif
