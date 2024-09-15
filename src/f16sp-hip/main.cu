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

#define NUM_OF_BLOCKS (1024 * 16)
#define NUM_OF_THREADS 128

__forceinline__ __device__ 
void reduceInShared_intrinsics(half2 * const v)
{
  int lid = threadIdx.x;  
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = __hadd2(v[lid], v[lid+i]);
    __syncthreads();
  }
}

__forceinline__ __device__
void reduceInShared_native(half2 * const v)
{
  int lid = threadIdx.x;  
  #pragma unroll
  for (int i = NUM_OF_THREADS/2; i >= 1; i = i / 2) {
    if(lid<i) v[lid] = v[lid] + v[lid+i];
    __syncthreads();
  }
}

__global__
void scalarProductKernel_intrinsics(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const int stride = gridDim.x*blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value = __float2half2_rn(0.f);

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
  {
    value = __hfma2(a[i], b[i], value);
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_intrinsics(shArray);

  if (threadIdx.x == 0)
  {
    half2 result = shArray[0];
    float f_result = __low2float(result) + __high2float(result);
    atomicAdd(results, f_result);
  }
}

__global__
void scalarProductKernel_native(
    half2 const *__restrict__ const a,
    half2 const *__restrict__ const b,
    float *__restrict__ const results,
    size_t const size)
{
  const int stride = gridDim.x*blockDim.x;
  __shared__ half2 shArray[NUM_OF_THREADS];

  half2 value(0.f, 0.f);
  shArray[threadIdx.x] = value;

  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
  {
    value += a[i] * b[i];
  }

  shArray[threadIdx.x] = value;
  __syncthreads();
  reduceInShared_native(shArray);

  if (threadIdx.x == 0)
  {
    half2 result = shArray[0];
    float f_result = (float)result.y + (float)result.x;
    atomicAdd(results, f_result);
  }
}

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

  float r1, r2, *d_r;

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

  float result_ref = 0.f;
  for (size_t i = 0; i < size; i++)
  {
    result_ref += (float)a[i].x * (float)b[i].x +
                  (float)a[i].y * (float)b[i].y;
  }

  // warmup
  for (int i = 0; i < repeat; i++)
    scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipMemset(d_r, 0, result_bytes);
    scalarProductKernel_intrinsics<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(&r1, d_r, result_bytes, hipMemcpyDeviceToHost);
  printf("Result (intrinsics)\t: %f \n", r1);

  // warmup
  for (int i = 0; i < repeat; i++)
    scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipMemset(d_r, 0, result_bytes);
    scalarProductKernel_native<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(d_a, d_b, d_r, size);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(&r2, d_r, result_bytes, hipMemcpyDeviceToHost);
  printf("Result (native operators)\t: %f \n", r2);

  bool ok = fabsf(r1 - result_ref) < 0.00001f &&
            fabsf(r2 - result_ref) < 0.00001f;
  printf("fp16ScalarProduct %s\n", ok ?  "PASS" : "FAIL");

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_r);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
