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
#include "common.h"

#define NUM_BLOCKS 1
#define BLOCK_SIZE 256

/* int64 atomic_min
inline
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
inline
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
inline
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
inline
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
inline
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
*/

// double atomic_max
inline
double atomic_max(double *address, double val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
    reinterpret_cast<unsigned long long*>(address)));

  const double old_val = *address;
  while(val > old_val) 
  {
    unsigned long long old = *reinterpret_cast<const unsigned long long*>(&old_val);
    if(obj.compare_exchange_strong(old, *reinterpret_cast<const unsigned long long*>(&val)))
      break;
  }
  return old_val;
}

template <typename T>
void atomicDerived (nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  //atomic_min(res, (T)i);
  atomic_max(res+1, (T)i);
}


int main() {
  unsigned long long res_u64[2] = {ULONG_MAX,0};
  long long res_s64[2] = {LONG_MAX,LONG_MIN};
  double res_f64[2] = {DBL_MAX,DBL_MIN};

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned long long, 1> d_res_u64 (2);   
  buffer<long long, 1> d_res_s64 (2);   
  buffer<double, 1> d_res_f64 (2);   

  range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  // the first two kernels should take almost the same execution time
  for (int n = 0; n < 1; n++) {
    q.submit([&] (handler &cgh) {
      auto acc = d_res_u64.get_access<sycl_write>(cgh);
      cgh.copy(res_u64, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_res_s64.get_access<sycl_write>(cgh);
      cgh.copy(res_s64, acc);
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_res_f64.get_access<sycl_write>(cgh);
      cgh.copy(res_f64, acc);
    });

    //atomicDerived<unsigned long long><<<NUM_BLOCKS, BLOCK_SIZE>>> (d_res_u64);
    //atomicDerived<long long><<<NUM_BLOCKS, BLOCK_SIZE>>> (d_res_s64);
    q.submit([&] (handler &cgh) {
      auto acc = d_res_f64.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class k3>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicDerived<double>(item, acc.get_pointer());
      });
    });
  }

  q.submit([&] (handler &cgh) {
    auto acc = d_res_u64.get_access<sycl_read>(cgh);
    cgh.copy(acc, res_u64);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_res_s64.get_access<sycl_read>(cgh);
    cgh.copy(acc, res_s64);
  });

  q.submit([&] (handler &cgh) {
    auto acc = d_res_f64.get_access<sycl_read>(cgh);
    cgh.copy(acc, res_f64);
  });

  q.wait();

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

  return 0;
}
