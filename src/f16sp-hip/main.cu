/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipcub/hipcub.hpp>
#include "kernels.h"

void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x = -1;
    temp.y = -1;
    a[i] = temp;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(half2);
  const size_t result_bytes = sizeof(float);

  half2 *a, *b;
  half2 *d_a, *d_b;

  float r1, r2, r3, r4, *d_r;

  a = (half2*) malloc (size_bytes);
  b = (half2*) malloc (size_bytes);
  hipMalloc((void**)&d_a, size_bytes);
  hipMalloc((void**)&d_b, size_bytes);

  hipMalloc((void**)&d_r, result_bytes);

  srand(123);
  generateInput(a, size);
  hipMemcpy(d_a, a, size_bytes, hipMemcpyHostToDevice);

  generateInput(b, size);
  hipMemcpy(d_b, b, size_bytes, hipMemcpyHostToDevice);

  double result_ref = 0.f;
  for (size_t i = 0; i < size; i++)
  {
    result_ref += (float)a[i].x * (float)b[i].x +
                  (float)a[i].y * (float)b[i].y;
  }

  // evaluate the grid sizes for performance optimization
  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid = grid / 2) {

    printf("GPU grid size is %d\n", grid);

    // warmup
    for (int i = 0; i < repeat; i++)
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_intrinsics<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r1, d_r, result_bytes, hipMemcpyDeviceToHost);

    // warmup
    for (int i = 0; i < repeat; i++)
      scalarProductKernel_native<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_native<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r2, d_r, result_bytes, hipMemcpyDeviceToHost);

    // warmup
    for (int i = 0; i < repeat; i++)
      scalarProductKernel_intrinsics2<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_intrinsics2<<<grid, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);
    hipMemcpy(&r3, d_r, result_bytes, hipMemcpyDeviceToHost);

    // warmup
    for (int i = 0; i < repeat; i++)
      scalarProductKernel_intrinsics3<<<grid, NUM_OF_THREADS>>>(
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b), d_r, size);

    hipDeviceSynchronize();
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_r, 0, result_bytes);
      scalarProductKernel_intrinsics3<<<grid, NUM_OF_THREADS>>>(
        reinterpret_cast<const float4*>(d_a),
        reinterpret_cast<const float4*>(d_b), d_r, size);
    }

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    hipMemcpy(&r4, d_r, result_bytes, hipMemcpyDeviceToHost);

#ifdef DEBUG
    printf("GPU Results: %f %f %f %f\n", r1, r2, r3, r4);
    printf("CPU Result: %f\n", (float)result_ref);
#endif

    bool ok = fabsf(r1 - (float)result_ref) < 0.00001f &&
              fabsf(r2 - (float)result_ref) < 0.00001f &&
              fabsf(r3 - (float)result_ref) < 0.00001f &&
              fabsf(r4 - (float)result_ref) < 0.00001f ;
    printf("fp16ScalarProduct %s\n\n", ok ? "PASS" : "FAIL");
  }

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_r);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
