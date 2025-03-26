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
void testKernel(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size,
    int N)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int gid = item.get_global_id(2);
      assert(gid < N) ;
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

// Performance impact of assert()
void perfKernel(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int gid = item.get_global_id(2);
      assert(gid <= item.get_local_range(2) * item.get_group_range(2));
      int s = 0;
      for (int n = 1; n <= gid; n++) {
        s++; assert(s <= gid);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void perfKernel2(
    sycl::queue &q,
    sycl::range<3> &gws,
    sycl::range<3> &lws,
    const int slm_size)
{
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int gid = item.get_global_id(2);
      int s = 0;
      for (int n = 1; n <= gid; n++) {
        s++; assert(s <= gid);
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
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
  sycl::range<3> gws (1, 1, Nblocks * Nthreads);
  sycl::range<3> lws (1, 1, Nthreads);

  printf("\nLaunch kernel to generate assertion failures\n");

  try {
    testKernel(q, gws, lws, 0, 60);

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

  sycl::range<3> gws (1, 1, Nblocks * Nthreads);
  sycl::range<3> lws (1, 1, Nthreads);

  printf("\nLaunch kernel to evaluate the impact of assertion on performance \n");

  printf("Each thread in the kernel executes threadID + 1 assertions\n");
  auto start = std::chrono::steady_clock::now();
  perfKernel(q, gws, lws, 0);
  q.wait();
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Kernel time : %f\n", time.count());

  printf("Each thread in the kernel executes threadID assertions\n");
  start = std::chrono::steady_clock::now();
  perfKernel2(q, gws, lws, 0);
  q.wait();
  end = std::chrono::steady_clock::now();
  time = end - start;
  printf("Kernel time : %f\n", time.count());

  return true;
}
