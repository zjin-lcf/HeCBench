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
void atomicMinDerived (nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_min(res, (T)i);
}

template <typename T>
void atomicMaxDerived (nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_max(res, (T)i);
}

template <typename T>
void atomicAddDerived (nd_item<1> &item, T *__restrict res)
{
  int i = item.get_global_id(0) + 1;
  atomic_add(res, (T)i);
}

template <typename T>
class k_atomic_min;

template <typename T>
class k_atomic_max;

template <typename T>
class k_atomic_add;

template <typename T>
void testMin (queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.memcpy(d_ptr, h_ptr, sizeof(T));
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class k_atomic_min<T>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicMinDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic min for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
}

template <typename T>
void testMax (queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.memcpy(d_ptr, h_ptr, sizeof(T));
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class k_atomic_max<T>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicMaxDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic max for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
}

template <typename T>
void testAdd (queue &q, T *h_ptr, T *d_ptr, const int repeat, const char* name) {
  range<1> gws (NUM_BLOCKS * BLOCK_SIZE);
  range<1> lws (BLOCK_SIZE);

  auto start = std::chrono::steady_clock::now();

  for (int n = 0; n < repeat; n++) {
    q.memcpy(d_ptr, h_ptr, sizeof(T));
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class k_atomic_add<T>>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        atomicAddDerived<T>(item, d_ptr);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Atomic add for data type %s | ", name);
  printf("Average execution time: %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(h_ptr, d_ptr, sizeof(T)).wait();
}


int main(int argc, char** argv) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned long long res_u64[3] = {ULONG_MAX,0,0};
  long long res_s64[3] = {LONG_MAX,LONG_MIN,0};
  double res_f64[3] = {DBL_MAX,DBL_MIN,0};

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  unsigned long long *d_res_u64 = sycl::malloc_device<unsigned long long>(3, q);
  long long *d_res_s64 = sycl::malloc_device<long long>(3, q);
  double *d_res_f64 = sycl::malloc_device<double>(3, q);

  testMin<unsigned long long>(q, res_u64, d_res_u64, repeat, "U64");
  testMin<long long>(q, res_s64, d_res_s64, repeat, "S64");
  testMin<double>(q, res_f64, d_res_f64, repeat, "F64");

  testMax<unsigned long long>(q, res_u64+1, d_res_u64+1, repeat, "U64");
  testMax<long long>(q, res_s64+1, d_res_s64+1, repeat, "S64");
  testMax<double>(q, res_f64+1, d_res_f64+1, repeat, "F64");

  // the add kernels are slow
  testAdd<unsigned long long>(q, res_u64+2, d_res_u64+2, 1, "U64");
  testAdd<long long>(q, res_s64+2, d_res_s64+2, 1, "S64");
  testAdd<double>(q, res_f64+2, d_res_f64+2, 1, "F64");

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

  sycl::free(d_res_u64, q);
  sycl::free(d_res_s64, q);
  sycl::free(d_res_f64, q);

  return 0;
}
