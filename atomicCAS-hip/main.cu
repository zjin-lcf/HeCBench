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
#include <hip/hip_runtime.h>

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

// int64 atomic add
__device__ __forceinline__
long long atomic_add(long long *address, long long val)
{
  long long old, newdbl, ret = *address;
  do {
    old = ret;
    newdbl = old+val;
  } while((ret = (long long)atomicCAS((unsigned long long*)address, (unsigned long long)old, (unsigned long long)newdbl)) != old);
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

// uint64 atomic add
__device__ __forceinline__
unsigned long long atomic_add(unsigned long long *address, unsigned long long val)
{
  unsigned long long old, newdbl, ret = *address;
  do {
    old = ret;
    newdbl = old+val;
  } while((ret = atomicCAS(address, old, newdbl)) != old);
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

// Double-precision floating point atomic add
__device__ __forceinline__
double atomic_add(double *address, double val)
{
  // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
  unsigned long long *ptr = (unsigned long long *)address;
  unsigned long long old, newdbl, ret = *ptr;
  do {
    old = ret;
    newdbl = __double_as_longlong(__longlong_as_double(old)+val);
  } while((ret = atomicCAS(ptr, old, newdbl)) != old);
  return __longlong_as_double(ret);
}


template <typename T>
__global__ 
void atomicDerived (T *__restrict res)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
  atomic_min(res, (T)i);
  atomic_max(res+1, (T)i);
  atomic_add(res+2, (T)i);
}


int main(int argc, char** argv) {

  const int repeat = atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX,0,0};
  long long res_s64[3] = {LONG_MAX,LONG_MIN,0};
  double res_f64[3] = {DBL_MAX,DBL_MIN,0};

  unsigned long long *d_res_u64;   
  long long *d_res_s64;   
  double *d_res_f64;   

  hipMalloc((void**)&d_res_u64, 3*sizeof(unsigned long long));
  hipMalloc((void**)&d_res_s64, 3*sizeof(long long));
  hipMalloc((void**)&d_res_f64, 3*sizeof(double));

  // the min/max kernels would take almost the same execution time for many iterations
  // the add kernels are very slow compared to min/max kernels
  for (int n = 0; n < repeat; n++) {
    hipMemcpy(d_res_u64, res_u64, 3*sizeof(unsigned long long), hipMemcpyHostToDevice);
    hipMemcpy(d_res_s64, res_s64, 3*sizeof(long long), hipMemcpyHostToDevice);
    hipMemcpy(d_res_f64, res_f64, 3*sizeof(double), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicDerived<unsigned long long>), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0, d_res_u64);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicDerived<long long>), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0, d_res_s64);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(atomicDerived<double>), dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0, d_res_f64);
  }

  hipMemcpy(&res_u64, d_res_u64, 3*sizeof(unsigned long long), hipMemcpyDeviceToHost);
  hipMemcpy(&res_s64, d_res_s64, 3*sizeof(long long), hipMemcpyDeviceToHost);
  hipMemcpy(&res_f64, d_res_f64, 3*sizeof(double), hipMemcpyDeviceToHost);

  unsigned long long bound = NUM_BLOCKS*BLOCK_SIZE;
  unsigned long long sum = 0; 
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
  if (res_u64[2] != sum || res_s64[2] != (long long)sum || res_f64[2] != (double)sum) {
    error = true;
    printf("atomic add results: %llu %lld %lf\n", res_u64[2], res_s64[2], res_f64[2]);
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  hipFree(d_res_u64);
  hipFree(d_res_s64);
  hipFree(d_res_f64);
  return 0;
}
