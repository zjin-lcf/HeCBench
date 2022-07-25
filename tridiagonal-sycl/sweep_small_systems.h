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
#include "common.h"
#include "tridiagonal.h"
#include "sweep_kernels.cpp"


const char *sweepKernelNames[] = { 
  "sweep_small_systems_local_kernel",      // use local memory for temp array
  "sweep_small_systems_global_kernel",    // use global memory for temp array
  "sweep_small_systems_global_vec4_kernel",  // use global memory abd solve 4 systems per thread
  "transpose",              // data reordering
};  

double runReorderKernel(queue &q, buffer<float,1> &dev_a, buffer<float,1> &dev_t, 
    int width, int height)
{
  size_t szGlobalWorkSize[2];
  size_t szLocalWorkSize[2];

  // set transpose kernel execution parameters
  szLocalWorkSize[0] = TRANSPOSE_BLOCK_DIM;
  szLocalWorkSize[1] = TRANSPOSE_BLOCK_DIM;
  szGlobalWorkSize[0] = shrRoundUp(TRANSPOSE_BLOCK_DIM, width);
  szGlobalWorkSize[1] = shrRoundUp(TRANSPOSE_BLOCK_DIM, height);
  range<2> gws (szGlobalWorkSize[1], szGlobalWorkSize[0]);
  range<2> lws (szLocalWorkSize[1], szLocalWorkSize[0]);

  // warmup
  q.submit([&] (handler &cgh) {
    auto d_t = dev_t.get_access<sycl_discard_write>(cgh);
    auto d_a = dev_a.get_access<sycl_read>(cgh);
    accessor<float, 1, sycl_read_write, access::target::local> 
    lmem(TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM+1), cgh);
    cgh.parallel_for<class transpose_array_warmup>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      transpose(item, d_t.get_pointer(), d_a.get_pointer(), lmem.get_pointer(), width, height);
    });
  });
  q.wait();

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    q.submit([&] (handler &cgh) {
      auto d_t = dev_t.get_access<sycl_discard_write>(cgh);
      auto d_a = dev_a.get_access<sycl_read>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> 
      lmem(TRANSPOSE_BLOCK_DIM * (TRANSPOSE_BLOCK_DIM+1), cgh);
      cgh.parallel_for<class transpose_array>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
          transpose(item, d_t.get_pointer(), d_a.get_pointer(), lmem.get_pointer(), width, height);
      });
    });
  }
  q.wait();
  sum_time = shrDeltaT(0);
  double time = sum_time / BENCH_ITERATIONS;

  return time;
}

double runSweepKernel(queue &q, 
    buffer<float, 1> &dev_a, 
    buffer<float, 1> &dev_b, 
    buffer<float, 1> &dev_c, 
    buffer<float, 1> &dev_d, 
    buffer<float, 1> &dev_x, 
    buffer<float, 1> &dev_t, 
    buffer<float, 1> &dev_w, 
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

  range<1> gws (szGlobalWorkSize);
  range<1> lws (szLocalWorkSize);

  // warm up
  if (useLmem) 
    q.submit([&] (handler &cgh) {
      auto a_d = dev_a.get_access<sycl_read>(cgh);
      auto b_d = dev_b.get_access<sycl_read>(cgh);
      auto c_d = dev_c.get_access<sycl_read>(cgh);
      auto d_d = dev_d.get_access<sycl_read>(cgh);
      auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class sweep_local_warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sweep_small_systems_local_kernel(
            item, 
            a_d.get_pointer(), 
            b_d.get_pointer(), 
            c_d.get_pointer(), 
            d_d.get_pointer(), 
            x_d.get_pointer(), 
            system_size,
            num_systems,
            reorder);
      });
    });

  else if (useVec4) 
    q.submit([&] (handler &cgh) {
      auto a_d = dev_a.get_access<sycl_read>(cgh);
      auto b_d = dev_b.get_access<sycl_read>(cgh);
      auto c_d = dev_c.get_access<sycl_read>(cgh);
      auto d_d = dev_d.get_access<sycl_read>(cgh);
      auto w_d = dev_w.get_access<sycl_discard_read_write>(cgh);
      auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class sweep_global_v4_warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sweep_small_systems_global_vec4_kernel(
            item, 
            a_d.get_pointer(), 
            b_d.get_pointer(), 
            c_d.get_pointer(), 
            d_d.get_pointer(), 
            x_d.get_pointer(), 
            w_d.get_pointer(), 
            system_size,
            num_systems,
            reorder);
      });
    });

  else 
    q.submit([&] (handler &cgh) {
      auto a_d = dev_a.get_access<sycl_read>(cgh);
      auto b_d = dev_b.get_access<sycl_read>(cgh);
      auto c_d = dev_c.get_access<sycl_read>(cgh);
      auto d_d = dev_d.get_access<sycl_read>(cgh);
      auto w_d = dev_w.get_access<sycl_discard_read_write>(cgh);
      auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
      cgh.parallel_for<class sweep_global_warmup>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        sweep_small_systems_global_kernel(
            item, 
            a_d.get_pointer(), 
            b_d.get_pointer(), 
            c_d.get_pointer(), 
            d_d.get_pointer(), 
            x_d.get_pointer(), 
            w_d.get_pointer(), 
            system_size,
            num_systems,
            reorder);
      });
    });

  q.wait();

  shrLog("  looping %i times..\n", BENCH_ITERATIONS);  

  // run computations on GPUs in parallel
  double sum_time = 0.0;
  shrDeltaT(0);
  for (int iCycles = 0; iCycles < BENCH_ITERATIONS; iCycles++)
  {
    if (useLmem) 
      q.submit([&] (handler &cgh) {
        auto a_d = dev_a.get_access<sycl_read>(cgh);
        auto b_d = dev_b.get_access<sycl_read>(cgh);
        auto c_d = dev_c.get_access<sycl_read>(cgh);
        auto d_d = dev_d.get_access<sycl_read>(cgh);
        auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class sweep_local>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          sweep_small_systems_local_kernel(
              item, 
              a_d.get_pointer(), 
              b_d.get_pointer(), 
              c_d.get_pointer(), 
              d_d.get_pointer(), 
              x_d.get_pointer(), 
              system_size,
              num_systems,
              reorder);
        });
      });

    else if (useVec4) 
      q.submit([&] (handler &cgh) {
        auto a_d = dev_a.get_access<sycl_read>(cgh);
        auto b_d = dev_b.get_access<sycl_read>(cgh);
        auto c_d = dev_c.get_access<sycl_read>(cgh);
        auto d_d = dev_d.get_access<sycl_read>(cgh);
        auto w_d = dev_w.get_access<sycl_discard_read_write>(cgh);
        auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class sweep_global_v4>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          sweep_small_systems_global_vec4_kernel(
              item, 
              a_d.get_pointer(), 
              b_d.get_pointer(), 
              c_d.get_pointer(), 
              d_d.get_pointer(), 
              x_d.get_pointer(), 
              w_d.get_pointer(), 
              system_size,
              num_systems,
              reorder);
        });
      });
    else 
      q.submit([&] (handler &cgh) {
        auto a_d = dev_a.get_access<sycl_read>(cgh);
        auto b_d = dev_b.get_access<sycl_read>(cgh);
        auto c_d = dev_c.get_access<sycl_read>(cgh);
        auto d_d = dev_d.get_access<sycl_read>(cgh);
        auto w_d = dev_w.get_access<sycl_discard_read_write>(cgh);
        auto x_d = dev_x.get_access<sycl_discard_write>(cgh);
        cgh.parallel_for<class sweep_global>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          sweep_small_systems_global_kernel(
              item, 
              a_d.get_pointer(), 
              b_d.get_pointer(), 
              c_d.get_pointer(), 
              d_d.get_pointer(), 
              x_d.get_pointer(), 
              w_d.get_pointer(), 
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

double sweep_small_systems(queue &q, float *a, float *b, float *c, float *d, float *x, 
    int system_size, int num_systems, bool reorder = false)
{
  if (reorder) shrLog("sweep_data_reorder_kernel\n"); 
  if (useLmem) shrLog("%s\n", sweepKernelNames[0]); 
  else if (useVec4) shrLog("%s\n", sweepKernelNames[2]); 
  else shrLog("%s\n", sweepKernelNames[1]); 

  const unsigned int mem_size = num_systems * system_size;

  buffer<float, 1> device_a (a, mem_size); 
  buffer<float, 1> device_b (b, mem_size); 
  buffer<float, 1> device_c (c, mem_size); 
  buffer<float, 1> device_d (d, mem_size); 
  //buffer<float, 1> device_x (x, mem_size); 
  buffer<float, 1> device_x (mem_size); 
  buffer<float, 1> device_t (mem_size); // additional array for reordering
  buffer<float, 1> device_w (mem_size); // global clone of private array

  int workSize = num_systems;

  double reorder_time = 0.0;
  double solver_time = 0.0;

  if (reorder)
  {
    // transpose input data
    reorder_time += runReorderKernel(q, device_a, device_t, system_size, workSize);
    std::swap(device_a, device_t);

    reorder_time += runReorderKernel(q, device_b, device_t, system_size, workSize);
    std::swap(device_b, device_t);

    reorder_time += runReorderKernel(q, device_c, device_t, system_size, workSize);
    std::swap(device_c, device_t);

    reorder_time += runReorderKernel(q, device_d, device_t, system_size, workSize);
    std::swap(device_d, device_t);
  }

  // run solver
  solver_time = runSweepKernel(q, device_a, device_b, device_c, device_d, 
      device_x, device_t, device_w, system_size, workSize, reorder);

  if (reorder)
  {
    // transpose result back
    reorder_time += runReorderKernel(q, device_x, device_t, workSize, system_size);
    std::swap(device_x, device_t);
  }

  // copy result from device to host explictly
  q.submit([&] (handler &cgh) {
    auto dev_x_acc = device_x.get_access<sycl_read>(cgh);
    cgh.copy(dev_x_acc, x);
  }).wait();

  return solver_time + reorder_time;
}

#endif
