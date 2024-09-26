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
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <numeric>
#include <execution>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include "shrUtils.h"

template <typename T>
__global__
void dot_product(const T *__restrict__ a,
                 const T *__restrict__ b,
                       T *__restrict__ d,
                 const size_t n)
{
  size_t iGID = blockIdx.x * blockDim.x + threadIdx.x;
  T sum = 0;
  for(size_t idx = iGID; idx < n; idx += gridDim.x * blockDim.x) {
    size_t iInOffset = idx * 4;
    sum += a[iInOffset    ] * b[iInOffset    ] +
           a[iInOffset + 1] * b[iInOffset + 1] +
           a[iInOffset + 2] * b[iInOffset + 2] +
           a[iInOffset + 3] * b[iInOffset + 3];
  }

  using BlockReduce = hipcub::BlockReduce<T, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  T aggregate = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) {
    atomicAdd(d, aggregate);
  }
}

template <typename T>
void dot (const size_t iNumElements, const int iNumIterations)
{
  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
  // rounded up to the nearest multiple of the LocalWorkSize
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(T);

  const size_t grid_size = shrRoundUp(szLocalWorkSize,
                                      szGlobalWorkSize / (szLocalWorkSize * 4));

  // Allocate and initialize host arrays
  T* srcA = (T*) malloc (src_size_bytes);
  T* srcB = (T*) malloc (src_size_bytes);
  T  dst;

  size_t i;
  srand(123);
  for (i = 0; i < iNumElements ; ++i)
  {
    srcA[i] = (i < iNumElements / 2) ? -1 : 1;
    srcB[i] = -1;
  }
  for (i = iNumElements; i < src_size ; ++i) {
    srcA[i] = srcB[i] = 0;
  }

  T *d_srcA;
  T *d_srcB;
  T *d_dst;

  hipMalloc((void**)&d_srcA, src_size_bytes);
  hipMemcpy(d_srcA, srcA, src_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_srcB, src_size_bytes);
  hipMemcpy(d_srcB, srcB, src_size_bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_dst, sizeof(T));

  dim3 grid (grid_size);
  dim3 block (szLocalWorkSize);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    hipMemset(d_dst, 0, sizeof(T));
    dot_product<<<grid, block>>>(d_srcA, d_srcB, d_dst, src_size / 4);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  hipMemcpy(&dst, d_dst, sizeof(T), hipMemcpyDeviceToHost);
  printf("%s\n\n", dst == T(0) ? "PASS" : "FAIL");

  hipblasHandle_t h;
  hipblasCreate(&h);
  hipblasSetPointerMode(h, HIPBLAS_POINTER_MODE_DEVICE);

  start = std::chrono::steady_clock::now();

  for (i = 0; i < (size_t)iNumIterations; i++) {
    hipDataType xType, yType, rType, eType;
    if constexpr (std::is_same<T, double>::value) {
      xType = yType = rType = eType = HIPBLAS_R_64F;
    } else if constexpr (std::is_same<T, float>::value) {
      xType = yType = rType = eType = HIPBLAS_R_32F;
    }

    hipblasDotEx(h, iNumElements, d_srcA, xType, 1, d_srcB,
                yType, 1, d_dst, rType, eType);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average hipblasDot execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);

  hipMemcpy(&dst, d_dst, sizeof(T), hipMemcpyDeviceToHost);
  printf("%s\n\n", dst == T(0) ? "PASS" : "FAIL");

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    dst = std::transform_reduce(std::execution::par_unseq,
                                d_srcA, d_srcA + iNumElements, d_srcB, .0);
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average std::transform_reduce execution time %f (ms)\n", (time * 1e-6f) / iNumIterations);
  printf("%s\n\n", dst == T(0) ? "PASS" : "FAIL");

  hipFree(d_dst);
  hipFree(d_srcA);
  hipFree(d_srcB);
  hipblasDestroy(h);

  free(srcA);
  free(srcB);
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  dot<float>(iNumElements, iNumIterations);
  dot<double>(iNumElements, iNumIterations);

  return EXIT_SUCCESS;
}
