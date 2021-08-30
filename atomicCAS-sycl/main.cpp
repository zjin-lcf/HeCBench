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

#define NUM_BLOCKS 1024
#define BLOCK_SIZE 256

// int64 atomic_min
long long atomic_min(long long *address, long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  long long old_val = *address;
  const unsigned long long new_val = *reinterpret_cast<const unsigned long long*>(&val);

  while(val < old_val) 
  {
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val))
      break;
  }
  return old_val;
}

// uint64 atomic_min
inline
unsigned long long atomic_min(unsigned long long *address, unsigned long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    (sycl::multi_ptr<unsigned long long, access::address_space::global_space>(address)));

  unsigned long long old_val = *address;
  while(val < old_val) 
  {
    if(obj.compare_exchange_strong(old_val, val))
      break;
  }
  return old_val;
}

// uint64 atomic add
inline
unsigned long long atomic_add(unsigned long long *address, unsigned long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj ((
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      address)));

  unsigned long long old_val = *address;
  while(true)
  {
    const unsigned long long new_val = old_val + val;
    if(obj.compare_exchange_strong(old_val, new_val))
      break;
  }
  return old_val;
}

// int64 atomic_max
long long atomic_max(long long *address, long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  long long old_val = *address;
  const unsigned long long new_val = *reinterpret_cast<const unsigned long long*>(&val);

  while(val > old_val) 
  {
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val))
      break;
  }
  return old_val;
}

// uint64 atomic_max
inline
unsigned long long atomic_max(unsigned long long *address, unsigned long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    (sycl::multi_ptr<unsigned long long, access::address_space::global_space>(address)));

  unsigned long long old_val = *address;
  while(val > old_val) 
  {
    if(obj.compare_exchange_strong(old_val, val))
      break;
  }
  return old_val;
}

// For all double atomics:
//      Must do the compare with integers, not floating point,
//      since NaN is never equal to any other NaN

// double atomic_min
double atomic_min(double *address, double val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  double old_val = *address;
  const unsigned long long new_val = *reinterpret_cast<const unsigned long long*>(&val);

  while(val < old_val) 
  {
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val))
      break;
  }
  return old_val;
}

// int64 atomic add
inline
long long atomic_add(long long *address, long long val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  long long old_val = *address;
  while(true)
  {
    const long long new_val = old_val + val;
    const unsigned long long new_val_u64 = 
      *reinterpret_cast<const unsigned long long*>(&new_val);
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val_u64))
      break;
  }
  return old_val;
}


// double atomic_max
inline
double atomic_max(double *address, double val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  double old_val = *address;
  const unsigned long long new_val = *reinterpret_cast<const unsigned long long*>(&val);

  while(val > old_val) 
  {
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val))
      break;
  }
  return old_val;
}

// Double-precision floating point atomic add
inline
double atomic_add(double *address, double val)
{
  sycl::atomic<unsigned long long, access::address_space::global_space> obj (
    sycl::multi_ptr<unsigned long long, access::address_space::global_space>(
      reinterpret_cast<unsigned long long*>(address)));

  double old_val = *address;
  while(true)
  {
    const double new_val = old_val + val;
    const unsigned long long new_val_u64 = 
      *reinterpret_cast<const unsigned long long*>(&new_val);
    if(obj.compare_exchange_strong(
       *reinterpret_cast<unsigned long long*>(&old_val), new_val_u64))
      break;
  }
  return old_val;
}

template <typename T>
void atomicDerived (nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_min(res, (T)i);
  atomic_max(res+1, (T)i);
  atomic_add(res+2, (T)i);
}


int main(int argc, char** argv) {

  const int repeat = atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX,0,0};
  long long res_s64[3] = {LONG_MAX,LONG_MIN,0};
  double res_f64[3] = {DBL_MAX,DBL_MIN,0};

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned long long, 1> d_res_u64 (3);   
  buffer<long long, 1> d_res_s64 (3);   
  buffer<double, 1> d_res_f64 (3);   

  range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  // the first two kernels should take almost the same execution time
  // the add kernels are very slow compared to min/max kernels
  for (int n = 0; n < repeat; n++) {
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

    q.submit([&] (handler &cgh) {
      auto acc = d_res_u64.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class k_ull>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicDerived<unsigned long long>(item, acc.get_pointer());
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_res_s64.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class k_ll>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicDerived<long long>(item, acc.get_pointer());
      });
    });

    q.submit([&] (handler &cgh) {
      auto acc = d_res_f64.get_access<sycl_read_write>(cgh);
      cgh.parallel_for<class k_double>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
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
  if (res_u64[2] != sum || res_s64[2] != (long long)sum || res_f64[2] != (double)sum) {
    error = true;
    printf("atomic add results: %llu %lld %lf\n", res_u64[2], res_s64[2], res_f64[2]);
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  return 0;
}
