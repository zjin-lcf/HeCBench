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

/*
 * This application demonstrates how to use the SYCL API to use multiple GPUs
 */

#include <math.h>
#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>

#ifndef MIN
#define MIN(a,b) (a < b ? a : b)
#endif

#include "simpleMultiDevice.h"

// Data configuration
#ifndef MAX_GPU_COUNT
  #define MAX_GPU_COUNT 8
#endif

const int DATA_N = 1048576 * 32;

// Simple reduction kernel.
// Refer to the 'reduction' CUDA Sample describing
// reduction optimization strategies

void reduceKernel(float *d_Result, const float *d_Input, int N,
                  sycl::nd_item<1> &item)
{
  const int tid = item.get_global_id(0);
  const int threadN = item.get_group_range(0) * item.get_local_range(0);
  float sum = 0;

  for (int pos = tid; pos < N; pos += threadN)
    sum += d_Input[pos];

  d_Result[tid] = sum;
}

// Program main
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  int repeat = atoi(argv[1]);

  //Solver config
  TGPUplan plan[MAX_GPU_COUNT];

  //GPU reduction results
  float h_SumGPU[MAX_GPU_COUNT];

  float sumGPU;
  double sumCPU, diff;

  int i, j, k, GPU_N;

  const int  BLOCK_N = 32;
  const int THREAD_N = 256;
  const int  ACCUM_N = BLOCK_N * THREAD_N;

  printf("Starting simpleMultiDevice\n");
  auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  GPU_N = gpu_devices.size();

  GPU_N = MIN(GPU_N, MAX_GPU_COUNT);

  printf("GPU device count: %i\n", GPU_N);

  printf("Generating input data of size %d ...\n\n", DATA_N);

  //Subdividing input data across GPUs
  //Get data sizes for each GPU
  for (i = 0; i < GPU_N; i++)
  {
    plan[i].dataN = DATA_N / GPU_N;
  }

  //Take into account "odd" data sizes
  for (i = 0; i < DATA_N % GPU_N; i++)
  {
    plan[i].dataN++;
  }

  //Assign data ranges to GPUs
  for (i = 0; i < GPU_N; i++)
  {
    plan[i].h_Sum = h_SumGPU + i;
  }

  //Create streams for issuing GPU command asynchronously
  //allocate memory (GPU and System page-locked)
  for (i = 0; i < GPU_N; i++)
  {
    plan[i].stream = sycl::queue(gpu_devices[i], sycl::property::queue::in_order());
    //Allocate memory
    plan[i].d_Data = sycl::malloc_device<float>(plan[i].dataN, plan[i].stream);
    plan[i].d_Sum = sycl::malloc_device<float>(ACCUM_N, plan[i].stream);
    plan[i].h_Sum_from_device = sycl::malloc_host<float>(ACCUM_N, plan[i].stream);
    plan[i].h_Data = sycl::malloc_host<float>(plan[i].dataN, plan[i].stream);

    for (j = 0; j < plan[i].dataN; j++)
    {
      plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
    }
  }

  //Start timing and compute on GPU(s)
  printf("Computing with %d GPUs...\n", GPU_N);

  sycl::range<1> gws (THREAD_N * BLOCK_N);
  sycl::range<1> lws (THREAD_N);

  auto start = std::chrono::steady_clock::now();

  for (k = 0; k < repeat; k++)
  {
    //Copy data to GPU, launch the kernel and copy data back. All asynchronously
    for (i = 0; i < GPU_N; i++)
    {
      //Copy input data from CPU
      plan[i].stream.memcpy(plan[i].d_Data, plan[i].h_Data,
                            plan[i].dataN * sizeof(float));

      //Perform GPU computations
      plan[i].stream.submit([&](sycl::handler &cgh) {
        auto plan_i_d_Sum = plan[i].d_Sum;
        auto plan_i_d_Data = plan[i].d_Data;
        auto plan_i_dataN = plan[i].dataN;
        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          reduceKernel(plan_i_d_Sum, plan_i_d_Data, plan_i_dataN, item);
        });
      });

      //Read back GPU results
      plan[i].stream.memcpy(plan[i].h_Sum_from_device, plan[i].d_Sum,
                            ACCUM_N * sizeof(float));
    }

    //Process GPU results
    for (i = 0; i < GPU_N; i++)
    {
      //Wait for all operations to finish
      plan[i].stream.wait();
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("  Average GPU Processing time: %f (us)\n\n", time * 1e-3f / repeat);

  for (i = 0; i < GPU_N; i++)
  {
    //Finalize GPU reduction for current subvector
    float sum = 0;

    for (j = 0; j < ACCUM_N; j++)
    {
      sum += plan[i].h_Sum_from_device[j];
    }

    *(plan[i].h_Sum) = sum;

    //Shut down this GPU
    sycl::free(plan[i].h_Sum_from_device, plan[i].stream);
    sycl::free(plan[i].d_Sum, plan[i].stream);
    sycl::free(plan[i].d_Data, plan[i].stream);
  }

  sumGPU = 0;

  for (i = 0; i < GPU_N; i++)
  {
    sumGPU += h_SumGPU[i];
  }

  // Compute on Host CPU
  printf("Computing with Host CPU...\n\n");

  sumCPU = 0;

  for (i = 0; i < GPU_N; i++)
  {
    for (j = 0; j < plan[i].dataN; j++)
    {
      sumCPU += plan[i].h_Data[j];
    }
  }

  // Compare GPU and CPU results
  printf("Comparing GPU and Host CPU results...\n");
  diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
  printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
  printf("  Relative difference: %E \n\n", diff);

  // Cleanup and shutdown
  for (i = 0; i < GPU_N; i++)
  {
    sycl::free(plan[i].h_Data, plan[i].stream);
  }

  exit((diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE);
}
