/*
 * Copyright 2010 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */


#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <float.h>
#include <cuda.h>

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

// int64 atomic_min
  __device__ __forceinline__
long long atomic_min(long long *address, long long val)
{
  long long ret = *address;
  while(val < ret)
  {
    long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
      break;
  }
  return ret;
}

// uint64 atomic_min
  __device__ __forceinline__
unsigned long long atomic_min(unsigned long long *address, unsigned long long val)
{
  unsigned long long ret = *address;
  while(val < ret)
  {
    unsigned long long old = ret;
    if((ret = atomicCAS(address, old, val)) == old)
      break;
  }
  return ret;
}

// int64 atomic_max
  __device__ __forceinline__
long long atomic_max(long long *address, long long val)
{
  long long ret = *address;
  while(val > ret)
  {
    long long old = ret;
    if((ret = (long long)atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
      break;
  }
  return ret;
}

// uint64 atomic_max
  __device__ __forceinline__
unsigned long long atomic_max(unsigned long long *address, unsigned long long val)
{
  unsigned long long ret = *address;
  while(val > ret)
  {
    unsigned long long old = ret;
    if((ret = atomicCAS(address, old, val)) == old)
      break;
  }
  return ret;
}

// For all double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN

// double atomic_min
  __device__ __forceinline__
double atomic_min(double *address, double val)
{
  unsigned long long ret = __double_as_longlong(*address);
  while(val < __longlong_as_double(ret))
  {
    unsigned long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}

// double atomic_max
  __device__ __forceinline__
double atomic_max(double *address, double val)
{
  unsigned long long ret = __double_as_longlong(*address);
  while(val > __longlong_as_double(ret))
  {
    unsigned long long old = ret;
    if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
      break;
  }
  return __longlong_as_double(ret);
}


template <typename T>
__global__ 
void atomicDerived (T *__restrict res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, (T)i);
  atomic_max(res+1, (T)i);
}


int main() {
  unsigned long long res_u64[2] = {ULONG_MAX,0};
  long long res_s64[2] = {LONG_MAX,LONG_MIN};
  double res_f64[2] = {DBL_MAX,DBL_MIN};

  unsigned long long *d_res_u64;   
  long long *d_res_s64;   
  double *d_res_f64;   

  cudaMalloc((void**)&d_res_u64, 2*sizeof(unsigned long long));
  cudaMalloc((void**)&d_res_s64, 2*sizeof(long long));
  cudaMalloc((void**)&d_res_f64, 2*sizeof(double));

  // the first two kernels should take almost the same execution time
  for (int n = 0; n < 5000; n++) {
    cudaMemcpy(d_res_u64, res_u64, 2*sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_s64, res_s64, 2*sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_f64, res_f64, 2*sizeof(double), cudaMemcpyHostToDevice);
    atomicDerived<unsigned long long><<<NUM_BLOCKS, BLOCK_SIZE>>> (d_res_u64);
    atomicDerived<long long><<<NUM_BLOCKS, BLOCK_SIZE>>> (d_res_s64);
    atomicDerived<double><<<NUM_BLOCKS, BLOCK_SIZE>>> (d_res_f64);
  }

  cudaMemcpy(&res_u64, d_res_u64, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res_s64, d_res_s64, 2*sizeof(long long), cudaMemcpyDeviceToHost);
  cudaMemcpy(&res_f64, d_res_f64, 2*sizeof(double), cudaMemcpyDeviceToHost);

  unsigned long long sum = 0; 
  unsigned long long bound = NUM_BLOCKS*BLOCK_SIZE;
  for (unsigned int i = 1; i <= bound; i++) sum += i;

  bool error = false;
  if (res_u64[0] != 1ULL || res_s64[0] != 1LL || res_f64[0] != 1.0) {
    error = true;
    printf("atomic min results: %llu %lld %lf\n", res_u64[0], res_s64[0], res_f64[0]);
  }
  if (res_u64[1] != bound || res_s64[1] != (long long)bound || res_f64[1] != (double)bound) {
    error = true;
    printf("atomic max results: %llu %lld %lf\n", res_u64[1], res_s64[1], res_f64[1]);
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  cudaFree(d_res_u64);
  cudaFree(d_res_s64);
  cudaFree(d_res_f64);
  return 0;
}
