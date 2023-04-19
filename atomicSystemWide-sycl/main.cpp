/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* A program demonstrating use of system-wide atomics on migratable memory.
 */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define min(a,b) (a) < (b) ? (a) : (b)
#define max(a,b) (a) > (b) ? (a) : (b)

using atomicSystemRef = sycl::atomic_ref<int,\
                        sycl::memory_order::relaxed,\
                        sycl::memory_scope::system,\
                        sycl::access::address_space::global_space>;

void atomicKernel(int *atom_arr, const int loop_num, sycl::nd_item<1> &item)
{
  int tid = item.get_global_id(0);

  for (int i=0; i < loop_num; i++)
  {
    // Atomic addition
    //atomicAdd_system(&atom_arr[0], 10);
    atomicSystemRef ar_add (atom_arr[0]);
    ar_add.fetch_add(10);

    // Atomic exchange
    //atomicExch_system(&atom_arr[1], tid);
    atomicSystemRef ar_exch (atom_arr[1]);
    ar_exch.exchange(tid);
 
    // Atomic maximum
    //atomicMax_system(&atom_arr[2], tid);
    atomicSystemRef ar_max (atom_arr[2]);
    ar_max.fetch_max(tid);

    // Atomic minimum
    //atomicMin_system(&atom_arr[3], tid);
    atomicSystemRef ar_min (atom_arr[3]);
    ar_min.fetch_min(tid);

    // Atomic increment (modulo 17+1)
    //atomicInc_system((unsigned int *)&atom_arr[4], 17);

    // Atomic decrement
    //atomicDec_system((unsigned int *)&atom_arr[5], 137);

    // Atomic compare-and-swap
    //atomicCAS_system(&atom_arr[6], tid - 1, tid);
    atomicSystemRef ar_cas (atom_arr[6]);
    int expected = tid - 1;
    ar_cas.compare_exchange_strong(expected, tid,
      sycl::memory_order::relaxed,
      sycl::memory_scope::system);

    // Bitwise atomic instructions

    // Atomic AND
    //atomicAnd_system(&atom_arr[7], 2 * tid + 7);
    atomicSystemRef ar_and (atom_arr[7]);
    ar_and.fetch_and(2 * tid + 7);

    // Atomic OR
    //atomicOr_system(&atom_arr[8], 1 << tid);
    atomicSystemRef ar_or (atom_arr[8]);
    ar_or.fetch_or(1 << tid);

    // Atomic XOR
    //atomicXor_system(&atom_arr[9], tid);
    atomicSystemRef ar_xor (atom_arr[9]);
    ar_xor.fetch_xor(tid);
  }
}

void atomicKernel_CPU(int *atom_arr, int no_of_threads, const int loop_num)
{

  for (int i=no_of_threads; i<2*no_of_threads; i++)
  {
    for (int j=0; j < loop_num; j++)
    {
      // Atomic addition
      __sync_fetch_and_add(&atom_arr[0],10);

      // Atomic exchange
      __sync_lock_test_and_set(&atom_arr[1], i);

      // Atomic maximum
      int old, expected;
      do {
        expected = atom_arr[2];
        old = __sync_val_compare_and_swap(&atom_arr[2], expected, max(expected, i));
      } while (old != expected);

      // Atomic minimum
      do {
        expected = atom_arr[3];
        old = __sync_val_compare_and_swap(&atom_arr[3], expected, min(expected, i));
      } while (old != expected);

      // Atomic increment (modulo 17+1)
      //int limit = 17;
      //do {
      //  expected = atom_arr[4];
      //  old = __sync_val_compare_and_swap(&atom_arr[4], expected, (expected >= limit) ? 0 : expected+1);
      //} while (old != expected);

      // Atomic decrement
      //limit = 137;
      //do {
      //  expected = atom_arr[5];
      //  old = __sync_val_compare_and_swap(&atom_arr[5], expected, ((expected == 0) || (expected > limit)) ? limit : expected-1);
      //} while (old != expected);

      // Atomic compare-and-swap
      __sync_val_compare_and_swap(&atom_arr[6], i-1, i);

      // Bitwise atomic instructions

      // Atomic AND
      __sync_fetch_and_and(&atom_arr[7] , 2 * i + 7);

      // Atomic OR
      __sync_fetch_and_or(&atom_arr[8], 1 << i);

      // Atomic XOR
      // 11th element should be 0xff
      __sync_fetch_and_xor(&atom_arr[9] ,i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
int verify(int *testData, const int len, const int loop_num)
{
  int val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val += 10;
  }

  if (val != testData[0])
  {
    printf("atomicAdd failed val = %d testData = %d\n", val, testData[0]);
    return false;
  }

  val = 0;

  bool found = false;

  for (int i = 0; i < len; ++i)
  {
    // second element should be a member of [0, len)
    if (i == testData[1])
    {
      found = true;
      break;
    }
  }

  if (!found)
  {
    printf("atomicExch failed\n");
    return false;
  }

  val = -(1 << 8);

  for (int i = 0; i < len; ++i)
  {
    // third element should be len-1
    val = max(val, i);
  }

  if (val != testData[2])
  {
    printf("atomicMax failed\n");
    return false;
  }

  val = 1 << 8;

  for (int i = 0; i < len; ++i)
  {
    val = min(val, i);
  }

  if (val != testData[3])
  {
    printf("atomicMin failed\n");
    return false;
  }

  /*
  int limit = 17;
  val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val = (val >= limit) ? 0 : val+1;
  }

  if (val != testData[4])
  {
    printf("atomicInc failed\n");
    return false;
  }

  limit = 137;
  val = 0;

  for (int i = 0; i < len*loop_num; ++i)
  {
    val = ((val == 0) || (val > limit)) ? limit : val-1;
  }

  if (val != testData[5])
  {
    printf("atomicDec failed\n");
    return false;
  }
  */

  found = false;

  for (int i = 0; i < len; ++i)
  {
    // seventh element should be a member of [0, len)
    if (i == testData[6])
    {
      found = true;
      break;
    }
  }

  if (!found)
  {
    printf("atomicCAS failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i)
  {
    // 8th element should be 1
    val &= (2 * i + 7);
  }

  if (val != testData[7])
  {
    printf("atomicAnd failed\n");
    return false;
  }

  val = 0;

  for (int i = 0; i < len; ++i)
  {
    // 9th element should be 0xff
    val |= (1 << i);
  }

  if (val != testData[8])
  {
    printf("atomicOr failed\n");
    return false;
  }

  val = 0xff;

  for (int i = 0; i < len; ++i)
  {
    // 11th element should be 0xff
    val ^= i;
  }

  if (val != testData[9])
  {
    printf("atomicXor failed\n");
    return false;
  }

  return true;
}


int main(int argc, char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <loop count within the kernel>\n", argv[0]);
    return 1;
  }
  const int loop_num = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto info = q.get_device().get_info<sycl::info::device::atomic_memory_scope_capabilities>();
  bool atomicMemoryScopeSystem = false;
  for (int i = 0; i < info.size(); i++) {
    if (info[i] == sycl::memory_scope::system) {
      atomicMemoryScopeSystem = true;
      break;
    }
  }

  if (!atomicMemoryScopeSystem) {
    // This sample requires being run on a device that supports system memory scope
    fprintf(stderr, "System memory scope not supported on this device. Exit\n");
    exit(1);
  }

  unsigned int numThreads = 256;
  unsigned int numBlocks = 64;
  unsigned int numData = 10;

  int *atom_arr = sycl::malloc_shared<int>(numData, q);

  // host memory or shared memory
  for (unsigned int i = 0; i < numData; i++)
    atom_arr[i] = 0;

  //To make the AND and XOR tests generate something other than 0...
  atom_arr[7] = atom_arr[9] = 0xff;

  sycl::range<1> gws (numBlocks * numThreads);
  sycl::range<1> lws (numThreads);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      atomicKernel(atom_arr, loop_num, item);
    });
  });
  atomicKernel_CPU(atom_arr, numBlocks*numThreads, loop_num);

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of atomic kernels on host and device: %f (s)\n", time * 1e-9f);

  // Compute & verify reference solution
  int testResult = verify(atom_arr, 2*numThreads*numBlocks, loop_num);

  sycl::free(atom_arr, q);

  printf("systemWideAtomics completed, returned %s \n", testResult ? "OK" : "ERROR!");
  exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
