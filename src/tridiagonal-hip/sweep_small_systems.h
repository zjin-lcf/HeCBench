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

#include <hip/hip_runtime.h>
#include <algorithm>   // std::swap
#include "tridiagonal.h"
#include "sweep_kernels.cu"


const char *sweepKernelNames[] = { 
  "sweep_small_systems_local_kernel",      // use local memory for temp array
  "sweep_small_systems_global_kernel",    // use global memory for temp array
  "sweep_small_systems_global_vec4_kernel",  // use global memory abd solve 4 systems per thread
  "transpose",              // data reordering
};  

double runReorderKernel(float* d_a, float* d_t, int width, int height)
{
  size_t szGlobalWorkSize[2];
  size_t szLocalWorkSize[2];

  // set transpose kernel execution parameters
  szLocalWorkSize[0] = TRANSPOSE_BLOCK_DIM;
  szLocalWorkSize[1] = TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[0] = shrRoundUp(TRANSPOSE_BLOCK_DIM, width) / TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[1] = shrRoundUp(TRANSPOSE_BLOCK_DIM, height) / TRANSPOSE_BLOCK_DIM;
  dim3 gws (szGlobalWorkSize[0], szGlobalWorkSize[1]);
  dim3 lws (szLocalWorkSize[0], szLocalWorkSize[1]);

  hipLaunchKernelGGL(transpose, gws, lws, 0, 0, d_t, d_a, width, height);

  hipDeviceSynchronize();

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    hipLaunchKernelGGL(transpose, gws, lws, 0, 0, d_t, d_a, width, height);
  }
  hipDeviceSynchronize();

  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double runSweepKernel(
    float* a_d, 
    float* b_d, 
    float* c_d, 
    float* d_d, 
    float* x_d, 
    float* t_d, 
    float* w_d, 
    int system_size, 
    int num_systems,
    bool reorder)
{
  size_t szGlobalWorkSize;
  size_t szLocalWorkSize;

  // set main kernel execution parameters
  if (useVec4) szLocalWorkSize = SWEEP_BLOCK_SIZE / 4;
  else szLocalWorkSize = SWEEP_BLOCK_SIZE;
  szGlobalWorkSize = shrRoundUp(SWEEP_BLOCK_SIZE, num_systems) / szLocalWorkSize;

  dim3 gws (szGlobalWorkSize);
  dim3 lws (szLocalWorkSize);

  // warm up
  if (useLmem) 
    hipLaunchKernelGGL(sweep_small_systems_local_kernel, gws, lws, 0, 0, 
      a_d, b_d, c_d, d_d, x_d, system_size, num_systems, reorder);
  else if (useVec4) 
    hipLaunchKernelGGL(sweep_small_systems_global_vec4_kernel, gws, lws, 0, 0, 
      a_d, b_d, c_d, d_d, x_d, w_d, system_size, num_systems, reorder);
  else 
    hipLaunchKernelGGL(sweep_small_systems_global_kernel, gws, lws, 0, 0, 
      a_d, b_d, c_d, d_d, x_d, w_d, system_size, num_systems, reorder);

  hipDeviceSynchronize();

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (useLmem) 
      hipLaunchKernelGGL(sweep_small_systems_local_kernel, gws, lws, 0, 0, 
        a_d, b_d, c_d, d_d, x_d, system_size, num_systems, reorder);
    else if (useVec4) 
      hipLaunchKernelGGL(sweep_small_systems_global_vec4_kernel, gws, lws, 0, 0, 
        a_d, b_d, c_d, d_d, x_d, w_d, system_size, num_systems, reorder);
    else 
      hipLaunchKernelGGL(sweep_small_systems_global_kernel, gws, lws, 0, 0, 
        a_d, b_d, c_d, d_d, x_d, w_d, system_size, num_systems, reorder);
  }

  hipDeviceSynchronize();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double sweep_small_systems(float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, bool reorder = false)
{
  if (reorder) shrLog("sweep_data_reorder_kernel\n"); 
  if (useLmem) shrLog("%s\n", sweepKernelNames[0]); 
  else if (useVec4) shrLog("%s\n", sweepKernelNames[2]); 
  else shrLog("%s\n", sweepKernelNames[1]); 

  const unsigned int mem_size = num_systems * system_size * sizeof(float);
  float* a_d;
  hipMalloc((void**)&a_d, mem_size);
  hipMemcpy(a_d, a, mem_size, hipMemcpyHostToDevice); 

  float* b_d;
  hipMalloc((void**)&b_d, mem_size);
  hipMemcpy(b_d, b, mem_size, hipMemcpyHostToDevice); 

  float* c_d;
  hipMalloc((void**)&c_d, mem_size); 
  hipMemcpy(c_d, c, mem_size, hipMemcpyHostToDevice);

  float* d_d;
  hipMalloc((void**)&d_d, mem_size); 
  hipMemcpy(d_d, d, mem_size, hipMemcpyHostToDevice);

  float* x_d;
  hipMalloc((void**)&x_d, mem_size); 

  float* t_d;
  hipMalloc((void**)&t_d, mem_size); 

  float* w_d;
  hipMalloc((void**)&w_d, mem_size); 

  int workSize = num_systems;

  double reorder_time = 0.0;
  double solver_time = 0.0;

  if (reorder)
  {
    // transpose input data
    reorder_time += runReorderKernel(a_d, t_d, system_size, workSize);
    std::swap(a_d, t_d);

    reorder_time += runReorderKernel(b_d, t_d, system_size, workSize);
    std::swap(b_d, t_d);

    reorder_time += runReorderKernel(c_d, t_d, system_size, workSize);
    std::swap(c_d, t_d);

    reorder_time += runReorderKernel(d_d, t_d, system_size, workSize);
    std::swap(d_d, t_d);
  }

  // run solver
  solver_time = runSweepKernel(a_d, b_d, c_d, d_d, 
      x_d, t_d, w_d, system_size, workSize, reorder);

  if (reorder)
  {
    // transpose result back
    reorder_time += runReorderKernel(x_d, t_d, workSize, system_size);
    std::swap(x_d, t_d);
  }
  
  // copy result from device to host 
  hipMemcpy(x, x_d, mem_size, hipMemcpyDeviceToHost); 

  hipFree(a_d);
  hipFree(b_d);
  hipFree(c_d);
  hipFree(d_d);
  hipFree(x_d);
  hipFree(t_d);
  hipFree(w_d);

  return solver_time + reorder_time;
}

#endif
