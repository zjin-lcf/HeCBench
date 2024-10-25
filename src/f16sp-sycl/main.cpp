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
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

void generateInput(sycl::half2 *a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    sycl::half2 temp;
    temp.x() = -1;
    temp.y() = -1;
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
  const size_t size_bytes = size * sizeof(sycl::half2);
  const size_t result_bytes = sizeof(float);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::half2 *a = (sycl::half2 *) malloc (size_bytes);
  sycl::half2 *b = (sycl::half2 *) malloc (size_bytes);
  float r1, r2, r3;

  float *d_r = sycl::malloc_device<float>(NUM_OF_BLOCKS, q);

  srand(123);
  generateInput(a, size);
  sycl::half2 *d_a = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_a, a, size_bytes);

  generateInput(b, size);
  sycl::half2 *d_b = sycl::malloc_device<sycl::half2>(size, q);
  q.memcpy(d_b, b, size_bytes);

  double result_ref = 0.f;
  for (size_t i = 0; i < size; i++)
  {
    result_ref += (float)a[i].x() * (float)b[i].x() +
                  (float)a[i].y() * (float)b[i].y();
  }

  // evaluate the grid sizes for performance optimization
  for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid = grid / 2) {

    printf("GPU grid size is %d\n", grid);

    // warmup
    for (int i = 0; i < repeat; i++) {
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::half2> shArray(sycl::range<1>(NUM_OF_THREADS), cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native(d_a, d_b, d_r, shArray.get_pointer(),
                                     size, item);
        });
      });
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<sycl::half2> shArray(sycl::range<1>(NUM_OF_THREADS), cgh);
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native(d_a, d_b, d_r, shArray.get_pointer(),
                                     size, item);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r1, d_r, result_bytes).wait();

    // warmup
    for (int i = 0; i < repeat; i++) {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native2(d_a, d_b, d_r, size, item);
        });
      });
    }
    q.wait();

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native2(d_a, d_b, d_r, size, item);
        });
      });
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r2, d_r, result_bytes).wait();

    // warmup
    for (int i = 0; i < repeat; i++) {
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native3(
            reinterpret_cast<const sycl::float4*>(d_a),
            reinterpret_cast<const sycl::float4*>(d_b),
            d_r, size, item);
        });
      });
    }
    q.wait();

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_r, 0, result_bytes);
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(grid*NUM_OF_THREADS),
                            sycl::range<1>(NUM_OF_THREADS)),
          [=](sycl::nd_item<1> item) {
          scalarProductKernel_native3(
            reinterpret_cast<const sycl::float4*>(d_a),
            reinterpret_cast<const sycl::float4*>(d_b),
            d_r, size, item);
        });
      });
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&r3, d_r, result_bytes).wait();

#ifdef DEBUG
    printf("GPU Results: %f %f %f\n", r1, r2, r3);
    printf("CPU Result: %f\n", (float)result_ref);
#endif

    bool ok = fabsf(r1 - (float)result_ref) < 0.00001f &&
              fabsf(r2 - (float)result_ref) < 0.00001f &&
              fabsf(r3 - (float)result_ref) < 0.00001f;
    printf("fp16ScalarProduct %s\n\n", ok ?  "PASS" : "FAIL");
  }

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_r, q);
  free(a);
  free(b);

  return EXIT_SUCCESS;
}
