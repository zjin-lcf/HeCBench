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

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Tridiagonal solvers.
 * Host code for sweep solver (one-system-per-thread).
 *
 * NVIDIA, Nikolai Sakharnykh, 2009
 */

#ifndef _SWEEP_SMALL_SYSTEMS_
#define _SWEEP_SMALL_SYSTEMS_

#include <algorithm>  // std::swap
#include <sycl/sycl.hpp>
#include "tridiagonal.h"
#include "sweep_kernels.cpp"


const char *sweepKernelNames[] = {
  "sweep_small_systems_local_kernel",      // use local memory for temp array
  "sweep_small_systems_global_kernel",    // use global memory for temp array
  "sweep_small_systems_global_vec4_kernel",  // use global memory abd solve 4 systems per thread
  "transpose",              // data reordering
};

double runReorderKernel(sycl::queue &q, float *d_a, float *d_t,
    int width, int height)
{
  size_t szGlobalWorkSize[2];
  size_t szLocalWorkSize[2];

  // set transpose kernel execution parameters
  szLocalWorkSize[0] = TRANSPOSE_BLOCK_DIM;
  szLocalWorkSize[1] = TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[0] = shrRoundUp(TRANSPOSE_BLOCK_DIM, width);
  szGlobalWorkSize[1] = shrRoundUp(TRANSPOSE_BLOCK_DIM, height);

  sycl::range<2> gws (szGlobalWorkSize[1], szGlobalWorkSize[0]);
  sycl::range<2> lws (szLocalWorkSize[1], szLocalWorkSize[0]);

  q.wait();

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> lmem (
         sycl::range<1>(TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM+1)), cgh);
      cgh.parallel_for<class transpose_array>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          transpose(item, d_t, d_a, lmem.get_pointer(), width, height);
      });
    });
  }
  q.wait();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double runSweepKernel(sycl::queue &q,
                      float *a_d,
                      float *b_d,
                      float *c_d,
                      float *d_d,
                      float *x_d,
                      float *t_d,
                      float *w_d,
    int system_size,
    int num_systems,
    bool reorder)
{
  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;

  // set main kernel execution parameters
  if (useVec4) szLocalWorkSize = SWEEP_BLOCK_SIZE / 4;
  else szLocalWorkSize = SWEEP_BLOCK_SIZE;
  szGlobalWorkSize = shrRoundUp(SWEEP_BLOCK_SIZE, num_systems);

  sycl::range<1> gws (szGlobalWorkSize);
  sycl::range<1> lws (szLocalWorkSize);

  q.wait();

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (useLmem)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class sweep_local>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          sweep_small_systems_local_kernel(
              item,
              a_d,
              b_d,
              c_d,
              d_d,
              x_d,
              system_size,
              num_systems,
              reorder);
        });
      });

    else if (useVec4)
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class sweep_global_v4>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          sweep_small_systems_global_vec4_kernel(
              item,
              a_d,
              b_d,
              c_d,
              d_d,
              x_d,
              w_d,
              system_size,
              num_systems,
              reorder);
        });
      });
    else
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class sweep_global>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          sweep_small_systems_global_kernel(
              item,
              a_d,
              b_d,
              c_d,
              d_d,
              x_d,
              w_d,
              system_size,
              num_systems,
              reorder);
        });
      });
  }

  q.wait();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double sweep_small_systems(sycl::queue &q, float *a, float *b, float *c, float *d, float *x,
    int system_size, int num_systems, bool reorder = false)
{
  if (reorder) shrLog("sweep_data_reorder_kernel\n");
  if (useLmem) shrLog("%s\n", sweepKernelNames[0]);
  else if (useVec4) shrLog("%s\n", sweepKernelNames[2]);
  else shrLog("%s\n", sweepKernelNames[1]);

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

  // additional array for reordering
  float *t_d = sycl::malloc_device<float>(mem_size, q);

  // global clone of private array
  float *w_d = sycl::malloc_device<float>(mem_size, q);

  int workSize = num_systems;

  double reorder_time = 0.0;
  double solver_time = 0.0;

  if (reorder)
  {
    // transpose input data
    reorder_time += runReorderKernel(q, a_d, t_d, system_size, workSize);
    std::swap(a_d, t_d);

    reorder_time += runReorderKernel(q, b_d, t_d, system_size, workSize);
    std::swap(b_d, t_d);

    reorder_time += runReorderKernel(q, c_d, t_d, system_size, workSize);
    std::swap(c_d, t_d);

    reorder_time += runReorderKernel(q, d_d, t_d, system_size, workSize);
    std::swap(d_d, t_d);
  }

  // run solver
  solver_time = runSweepKernel(q, a_d, b_d, c_d, d_d,
      x_d, t_d, w_d, system_size, workSize, reorder);

  if (reorder)
  {
    // transpose result back
    reorder_time += runReorderKernel(q, x_d, t_d, workSize, system_size);
    std::swap(x_d, t_d);
  }

  // copy result from device to host
  q.memcpy(x, x_d, mem_size * sizeof(float)).wait();

  sycl::free(a_d, q);
  sycl::free(b_d, q);
  sycl::free(c_d, q);
  sycl::free(d_d, q);
  sycl::free(x_d, q);
  sycl::free(t_d, q);
  sycl::free(w_d, q);

  return solver_time + reorder_time;
}

#endif
