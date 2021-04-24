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
 * Host code for cyclic reduction (CR).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _CYCLIC_SMALL_SYSTEMS_
#define _CYCLIC_SMALL_SYSTEMS_

#include <omp.h>
#include "tridiagonal.h"
#include "cyclic_kernels.cpp"

const char *cyclicKernelNames[] = { 
  "cyclic_small_systems_kernel",    // original version
  "cyclic_branch_free_kernel",      // optimized branch-free version
};  

double cyclic_small_systems(float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, int id = 0)
{
  shrLog(" %s\n", cyclicKernelNames[id]);

  const unsigned int mem_size = num_systems * system_size;
  double sum_time;

  #pragma omp target data map(to: a[0:mem_size], \
                                  b[0:mem_size], \
                                  c[0:mem_size], \
                                  d[0:mem_size]) \
                          map(from: x[0:mem_size])
{

  size_t szTeams;
  size_t szThreads;
  int iterations = my_log2 (system_size/2);

  // set execution parameters
  szThreads = system_size / 2;
  szTeams = num_systems; 

  if (id == 0)
    cyclic_small_systems_kernel(
        a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
  else
    cyclic_branch_free_kernel(
        a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (id == 0)
      cyclic_small_systems_kernel(
        a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
    else
      cyclic_branch_free_kernel(
        a, b, c, d, x, system_size, num_systems, iterations, szTeams, szThreads);
  }
  sum_time = shrDeltaT(0);

}

  double time = sum_time / BENCH_ITERATIONS;
  return time;
}

#endif
