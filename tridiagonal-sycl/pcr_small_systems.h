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

#include "common.h"
#include "tridiagonal.h"
#include "pcr_kernels.cpp"

const char *pcrKernelNames[] = { 
  "pcr_small_systems_kernel",    // original version
  "pcr_branch_free_kernel",      // optimized branch-free version
};  

double pcr_small_systems(queue &q, float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, int id = 0)
{
  shrLog(" %s\n", pcrKernelNames[id]);

  const unsigned int mem_size = num_systems * system_size;
  buffer<float, 1> device_a (a, mem_size); 
  buffer<float, 1> device_b (b, mem_size); 
  buffer<float, 1> device_c (c, mem_size); 
  buffer<float, 1> device_d (d, mem_size); 
  buffer<float, 1> device_x (x, mem_size); 

  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;
  int iterations = log2(system_size/2);

  // set execution parameters
  szLocalWorkSize = system_size;
  szGlobalWorkSize = num_systems * szLocalWorkSize;

  range<1> gws (szGlobalWorkSize);
  range<1> lws (szLocalWorkSize);

  // warm up
  if (id == 0)
    q.submit([&] (handler &cgh) {
      auto a_d = device_a.get_access<sycl_read>(cgh);
      auto b_d = device_b.get_access<sycl_read>(cgh);
      auto c_d = device_c.get_access<sycl_read>(cgh);
      auto d_d = device_d.get_access<sycl_read>(cgh);
      auto x_d = device_x.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> lmem((system_size+1)*5, cgh);
      cgh.parallel_for<class pcr_warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pcr_small_systems_kernel(item, 
            a_d.get_pointer(),
            b_d.get_pointer(),
            c_d.get_pointer(),
            d_d.get_pointer(),
            x_d.get_pointer(),
            lmem.get_pointer(),
            system_size, 
            num_systems, 
            iterations);
      });
    });
  else
    q.submit([&] (handler &cgh) {
      auto a_d = device_a.get_access<sycl_read>(cgh);
      auto b_d = device_b.get_access<sycl_read>(cgh);
      auto c_d = device_c.get_access<sycl_read>(cgh);
      auto d_d = device_d.get_access<sycl_read>(cgh);
      auto x_d = device_x.get_access<sycl_discard_write>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> lmem((system_size+1)*5, cgh);
      cgh.parallel_for<class pcr_opt_warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        pcr_branch_free_kernel(item, 
            a_d.get_pointer(),
            b_d.get_pointer(),
            c_d.get_pointer(),
            d_d.get_pointer(),
            x_d.get_pointer(),
            lmem.get_pointer(),
            system_size, 
            num_systems, 
            iterations);
      });
    });

  q.wait();
  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (id == 0)
      q.submit([&] (handler &cgh) {
        auto a_d = device_a.get_access<sycl_read>(cgh);
        auto b_d = device_b.get_access<sycl_read>(cgh);
        auto c_d = device_c.get_access<sycl_read>(cgh);
        auto d_d = device_d.get_access<sycl_read>(cgh);
        auto x_d = device_x.get_access<sycl_discard_write>(cgh);
        accessor<float, 1, sycl_read_write, access::target::local> lmem((system_size+1)*5, cgh);
        cgh.parallel_for<class pcr>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          pcr_small_systems_kernel(item, 
              a_d.get_pointer(),
              b_d.get_pointer(),
              c_d.get_pointer(),
              d_d.get_pointer(),
              x_d.get_pointer(),
              lmem.get_pointer(),
              system_size, 
              num_systems, 
              iterations);
        });
      });
    else
      q.submit([&] (handler &cgh) {
        auto a_d = device_a.get_access<sycl_read>(cgh);
        auto b_d = device_b.get_access<sycl_read>(cgh);
        auto c_d = device_c.get_access<sycl_read>(cgh);
        auto d_d = device_d.get_access<sycl_read>(cgh);
        auto x_d = device_x.get_access<sycl_discard_write>(cgh);
        accessor<float, 1, sycl_read_write, access::target::local> lmem((system_size+1)*5, cgh);
        cgh.parallel_for<class pcr_opt>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          pcr_branch_free_kernel(item, 
              a_d.get_pointer(),
              b_d.get_pointer(),
              c_d.get_pointer(),
              d_d.get_pointer(),
              x_d.get_pointer(),
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
  return time;

  // implicit write-back to the array x
}
#endif
