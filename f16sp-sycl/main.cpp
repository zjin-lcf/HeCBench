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
#include <cmath>
#include "common.h"

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 128

inline
void reduceInShared_native(half2 *const v, nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  if(lid<64) v[lid] = v[lid] + v[lid+64];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+32];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+16];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+8];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+4];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+2];
  item.barrier(access::fence_space::local_space);
  if(lid<32) v[lid] = v[lid] + v[lid+1];
  item.barrier(access::fence_space::local_space);
}

void scalarProductKernel_intrinsics(const half2 *a,
                                    const half2 *b,
                                    float *results,
                                          half2 *shArray,
                                    const size_t size,
                                    nd_item<1> &item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0); 

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  half2 value = float2{0.f, 0.f}.convert<half, rounding_mode::rte>();

  for (int i = item.get_global_id(0); i < size; i += stride)
  {
    value = fma(a[i], b[i], value);
  }

  shArray[lid] = value;
  item.barrier(access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    half2 result = shArray[0];
    float f_result = result[1] + result[0];
    results[gid] = f_result;
  }
}

void scalarProductKernel_native(const half2 *a,
                                const half2 *b,
                                float *results, 
                                      half2 *shArray,
                                const size_t size,
                                nd_item<1> item)
{
  int lid = item.get_local_id(0);
  int gid = item.get_group(0); 

  const int stride = item.get_group_range(0) * item.get_local_range(0);

  half2 value(0.f, 0.f);
  shArray[lid] = value;

  for (int i = item.get_global_id(0); i < size; i += stride)
  {
    value = a[i] * b[i] + value;
  }

  shArray[lid] = value;
  item.barrier(access::fence_space::local_space);
  reduceInShared_native(shArray, item);

  if (lid == 0)
  {
    half2 result = shArray[0];
    float f_result = (float)result.y() + (float)result.x();
    results[gid] = f_result;
  }
}

void generateInput(half2 *a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x() = static_cast<float>(rand() % 4);
    temp.y() = static_cast<float>(rand() % 2);
    a[i] = temp;
  }
}

int main(int argc, char *argv[])
{
  size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS*16;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  half2 *a = (half2 *)malloc(size * sizeof(half2));
  half2 *b = (half2 *)malloc(size * sizeof(half2));
  float *r = (float*) malloc (NUM_OF_BLOCKS*sizeof(float));
  buffer<float, 1> d_r (NUM_OF_BLOCKS);

  srand(123); 
  generateInput(a, size);
  buffer<half2, 1> d_a (a, size);

  generateInput(b, size);
  buffer<half2, 1> d_b (b, size);

  range<1> gws (NUM_OF_BLOCKS * NUM_OF_THREADS);
  range<1> lws (NUM_OF_THREADS);

  for (int i = 0; i < 10000; i++)
    q.submit([&](handler &cgh) {
      auto a = d_a.get_access<sycl_read>(cgh); 
      auto b = d_b.get_access<sycl_read>(cgh); 
      auto r = d_r.get_access<sycl_discard_write>(cgh); 
      accessor<half2, 1, sycl_read_write, sycl_lmem> shArray(NUM_OF_THREADS, cgh);
      cgh.parallel_for<class sp1>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        scalarProductKernel_intrinsics(
            a.get_pointer(),
            b.get_pointer(),
            r.get_pointer(),
            shArray.get_pointer(),
            size, item);
        });
      });

  q.submit([&](handler &cgh) {
    auto acc = d_r.get_access<sycl_read>(cgh); 
    cgh.copy(acc, r);
  }).wait();

  float result_intrinsics = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i)
  {
    result_intrinsics += r[i];
  }
  printf("Result intrinsics\t: %f \n", result_intrinsics);

  for (int i = 0; i < 10000; i++)
    q.submit([&](handler &cgh) {
      auto a = d_a.get_access<sycl_read>(cgh); 
      auto b = d_b.get_access<sycl_read>(cgh); 
      auto r = d_r.get_access<sycl_discard_write>(cgh); 
      accessor<half2, 1, sycl_read_write, sycl_lmem> shArray(NUM_OF_THREADS, cgh);
      cgh.parallel_for<class sp2>(nd_range<1>(gws, lws), [=](nd_item<1> item) {
        scalarProductKernel_native(
          a.get_pointer(),
          b.get_pointer(),
          r.get_pointer(),
          shArray.get_pointer(),
          size, item);
        });
      });

  q.submit([&](handler &cgh) {
    auto acc = d_r.get_access<sycl_read>(cgh); 
    cgh.copy(acc, r);
  }).wait();

  float result_native = 0;
  for (int i = 0; i < NUM_OF_BLOCKS; ++i)
  {
    result_native += r[i];
  }
  printf("Result native operators\t: %f \n", result_native);

  printf("fp16ScalarProduct %s\n", (fabs(result_intrinsics - result_native) < 0.00001) ? 
      "PASSED" : "FAILED");

  free(a);
  free(b);
  free(r);

  return EXIT_SUCCESS;
}
