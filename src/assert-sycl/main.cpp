/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <exception>
#include <sycl/sycl.hpp>

auto report_error = [] (sycl::exception_list elist) {
  for (auto &e : elist) {
    try { std::rethrow_exception(e); }
    catch (sycl::exception& e) {
      std::cerr << "Error:\n" << e.what() << std::endl;
    }
  }
};

// Tests assert function.
// Thread whose id > N will print assertion failed error message.
void testKernel(int N, sycl::nd_item<1> &item)
{
  int gid = item.get_global_id(0);
  assert(gid < N) ;
}

// Performance impact of assert()
void perfKernel(sycl::nd_item<1> &item)
{
  int gid = item.get_global_id(0);
  assert(gid <= item.get_local_range(0) * item.get_group_range(0));
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++; assert(s <= gid);
  }
}

void perfKernel2(sycl::nd_item<1> &item)
{
  int gid = item.get_global_id(0);
  int s = 0;
  for (int n = 1; n <= gid; n++) {
    s++; assert(s <= gid);
  }
}

// Declaration, forward
bool runPerf(sycl::queue &q, int argc, char **argv);
bool runTest(sycl::queue &q, int argc, char **argv);

int main(int argc, char **argv)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, report_error, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, report_error, sycl::property::queue::in_order());
#endif

  // The test expects no assertError
  runPerf(q, argc, argv);

  // The test expects assertError
  bool testResult = runTest(q, argc, argv);

  printf("Test assert completed, returned %s\n",
         testResult ? "OK" : "ERROR!");

  if (!testResult) return EXIT_FAILURE;

  exit(EXIT_SUCCESS);
}

bool runTest(sycl::queue &q, int argc, char **argv) {
  int Nblocks = 2;
  int Nthreads = 32;

  // Kernel configuration, where a one-dimensional
  // grid and one-dimensional blocks are configured.
  sycl::range<1> gws (Nblocks * Nthreads);
  sycl::range<1> lws (Nthreads);

  printf("\nLaunch kernel to generate assertion failures\n");

  try {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        testKernel(60, item);
      });
    });

    // Synchronize (flushes assert output).
    printf("\n-- Begin assert output\n\n");
    q.wait_and_throw();
    printf("\n-- End assert output\n\n");
  }
  catch (...) {}
  return true;
}

bool runPerf(sycl::queue &q, int argc, char **argv)
{
  int Nblocks = 1000;
  int Nthreads = 256;

  sycl::range<1> gws (Nblocks * Nthreads);
  sycl::range<1> lws (Nthreads);

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      perfKernel(item);
    });
  }).wait();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      perfKernel2(item);
    });
  }).wait();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}
