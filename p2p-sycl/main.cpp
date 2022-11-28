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

/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include "common.h"

void SimpleKernel(nd_item<1> &item, const float *src, float *dst)
{
  // Just a dummy kernel, doing enough for us to verify that everything
  // worked
  const int idx = item.get_global_id(0);
  dst[idx] = src[idx] * 2.0f;
}

inline bool IsAppBuiltAs64()
{
  return sizeof(void*) == 8;
}

int main(int argc, char **argv)
{
  printf("[%s] - Starting...\n", argv[0]);
  const int repeat = atoi(argv[1]);

  if (!IsAppBuiltAs64())
  {
    printf("%s is only supported with on 64-bit OSs and the application must be built as a 64-bit target.  Test is being waived.\n", argv[0]);
    exit(0);
  }

  printf("Checking for multiple GPUs...\n");
  std::vector<sycl::device> Devs;
  // look through all platforms (https://github.com/intel/llvm/issues/6749)
  for (const auto &platform : platform::get_platforms()) {
    // if it is a cuda "platform" then add the device to the list
    if (platform.get_backend() == backend::cuda)
      Devs.push_back(platform.get_devices()[0]);
  }

  if (Devs.size() < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Waiving test.\n");
    exit(0);
  }

  printf("There are %d GPUs\n", Devs.size());

  // create queues as desired
  auto q0 = queue{Devs[0], property::queue::in_order()}; // this corresponds to the lowest cuda ID available (usually 0)
  auto q1 = queue{Devs[1], property::queue::in_order()}; // this corresponds to the second lowest cuda ID available (usually 1)

  // hardcoded
  int gpuid[2] = {0, 1};

  // Allocate buffers
  const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);

  // GPU0
  float *g0 = (float*) malloc_device(buf_size, q0);
  float *h0 = (float*) malloc_host(buf_size, q0);

  // GPU1
  float *g1 = (float*) malloc_device(buf_size, q1);

  q0.wait();
  q1.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i=0; i<repeat; i++)
  {
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      q0.memcpy(g1, g0, buf_size).wait();
    }
    else
    {
      q1.memcpy(g0, g1, buf_size).wait();
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time_memcpy = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  printf("memcpy between GPU%d and GPU%d: %.2fGB/s\n", gpuid[0], gpuid[1],
      1.0f / time_memcpy * (repeat * buf_size));

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

  const int buf_len = buf_size / sizeof(float);
  for (int i=0; i<buf_len; i++)
  {
    h0[i] = float(i % 4096);
  }

  q0.memcpy(g0, h0, buf_size).wait();

  // Kernel launch configuration
  range<1> lws(256);
  range<1> gws(buf_len / 256 * 256);

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
  // output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
      gpuid[1], gpuid[0], gpuid[1]);

  q1.submit([&] (handler &cgh) {
    cgh.parallel_for<class k1>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      SimpleKernel(item, g0, g1);
    });
  });

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
  // output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to GPU%d...\n",
      gpuid[0], gpuid[1], gpuid[0]);

  q0.submit([&] (handler &cgh) {
    cgh.parallel_for<class k2>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      SimpleKernel(item, g1, g0);
    });
  });

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
  q0.memcpy(h0, g0, buf_size).wait();

  int error_count = 0;

  for (int i=0; i<buf_len; i++)
  {
    // Re-generate input data and apply 2x '* 2.0f' computation of both
    // kernel runs
    if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i%4096)*2.0f*2.0f));

      if (error_count++ > 10)
      {
        break;
      }
    }
  }

  // Cleanup and shutdown
  printf("Shutting down...\n");
  free(g0, q0);
  free(h0, q0);
  free(g1, q1);

  if (error_count != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }
  else
  {
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
  }
}
