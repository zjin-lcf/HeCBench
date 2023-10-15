/* -----------------------------------------------------------------------------
 * Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 * See 'LICENSE' in the project root for license information.
 * -------------------------------------------------------------------------- */
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <sycl/sycl.hpp>

#define NUM_GROUPS 1
#define GROUP_SIZE 1
#define WARMUP_RUN_COUNT 100
#define TIMING_RUN_COUNT 1000
#define TOTAL_RUN_COUNT WARMUP_RUN_COUNT + TIMING_RUN_COUNT
#define BATCH_SIZE 1000


void print_timing(std::string test, const std::array<float, TOTAL_RUN_COUNT> &results, int batch = 1)
{
  float total_us = 0.0f, mean_us = 0.0f, stddev_us = 0.0f;

  // skip warm-up runs
  auto start_iter = std::next(results.begin(), WARMUP_RUN_COUNT);
  auto end_iter = results.end();

  // mean
  std::for_each(start_iter, end_iter, [&](const float &run_ms) {
    total_us += (run_ms * 1000) / batch;
  });
  mean_us = total_us  / TIMING_RUN_COUNT;

  // stddev
  total_us = 0;
  std::for_each(start_iter, end_iter, [&](const float &run_ms) {
    float dev_us = ((run_ms * 1000) / batch) - mean_us;
    total_us += dev_us * dev_us;
  });
  stddev_us = sqrt(total_us / TIMING_RUN_COUNT);

  // display
  printf("\n %s: mean = %.1f us, stddev = %.1f us\n", test.c_str(), mean_us, stddev_us);
}

int main() {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::array<float, TOTAL_RUN_COUNT> results;

  sycl::range<1> gws (NUM_GROUPS * GROUP_SIZE);
  sycl::range<1> lws (GROUP_SIZE);

  //------------------------------------------------------------------------------------
  // Kernel launch enqueue rate:
  // Measure time taken to enqueue a kernel on the GPU
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      });
    });
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Enqueue rate", results);

  //------------------------------------------------------------------------------------
  // Single dispatch execution latency
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      });
    }).wait();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Single dispatch latency", results);

  //------------------------------------------------------------------------------------
  // Batch dispatch execution latency
  //------------------------------------------------------------------------------------
  for (auto i = 0; i < TOTAL_RUN_COUNT; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < BATCH_SIZE; j++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        });
      });
    }
    q.wait();
    auto stop = std::chrono::high_resolution_clock::now();
    results[i] = std::chrono::duration<float, std::milli>(stop - start).count();
  }
  print_timing("Batch dispatch latency", results, BATCH_SIZE);

  return 0;
}
