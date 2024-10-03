/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

int main(int argc, char** argv)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int arrayLength = 52428800;
  int block_sizes[] = {128, 256, 512, 1024};
  int N = 32;

  if (argc == 3) {
    arrayLength=atoi(argv[1]);
    N=atoi(argv[2]);
  }

  std::cout << "Array size: " << arrayLength*sizeof(int)/1024.0/1024.0 << " MB"<<std::endl;
  std::cout << "Repeat the kernel execution: " << N << " times" << std::endl;

  int* array=(int*)malloc(arrayLength*sizeof(int));
  int checksum =0;
  for(int i=0;i<arrayLength;i++) {
    array[i]=rand()%2;
    checksum+=array[i];
  }

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  size_t size=sizeof(int)*arrayLength;

  // Get device properties
  auto device = q.get_device();
  auto deviceName = device.get_info<sycl::info::device::name>();
  std::cout << "Device name: " << deviceName << std::endl;

  int *in, *out;
  in = sycl::malloc_device<int>(arrayLength, q);
  out = sycl::malloc_device<int>(1, q);

  q.memcpy(in, array, size).wait();

  // warmup
  for(int i=0;i<N;i++) {
    q.memset(out, 0, sizeof(int)).wait();
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(2048*256),
                                     sycl::range<1>(256)),
      [=](sycl::nd_item<1> item) {
      atomic_reduction(in, out, arrayLength, item);
    });
  }
  q.wait();

  float GB=(float)arrayLength*sizeof(int)*N;
  int sum;
  double times;

#define benchmark(kernel_name, grid_size)                                      \
  t1 = std::chrono::high_resolution_clock::now();                              \
  for (int i = 0; i < N; i++) {                                                \
    q.memset(out, 0, sizeof(int));                                             \
    q.submit([&](sycl::handler &cgh) {                                         \
      cgh.parallel_for(sycl::nd_range<1>(                                      \
        sycl::range<1>(block_size * grid_size),                                \
        sycl::range<1>(block_size)), [=](sycl::nd_item<1> item) {              \
        kernel_name(in, out, arrayLength, item);                               \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  q.wait();                                                                    \
  t2 = std::chrono::high_resolution_clock::now();                              \
  times = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)   \
              .count();                                                        \
  std::cout << "Thread block size: " << block_size << ", ";                    \
  std::cout << "The average performance of reduction is "                      \
            << 1.0E-09 * GB / times << " GBytes/sec" << std::endl;             \
  q.memcpy(&sum, out, sizeof(int)).wait();                                     \
  if (sum == checksum)                                                         \
    std::cout << "VERIFICATION: PASS" << std::endl << std::endl;               \
  else                                                                         \
    std::cout << "VERIFICATION: FAIL!!" << std::endl << std::endl;

  for (size_t k = 0; k < sizeof(block_sizes) / sizeof(int); k++) {
    int block_size = block_sizes[k];
    int blocks=std::min((arrayLength+block_size-1)/block_size,2048);
    benchmark(atomic_reduction, blocks);
    benchmark(atomic_reduction_v2, blocks / 2);
    benchmark(atomic_reduction_v4, blocks / 4);
    benchmark(atomic_reduction_v8, blocks / 8);
    benchmark(atomic_reduction_v16, blocks / 16);
  }

  sycl::free(in, q);
  sycl::free(out, q);
  free(array);
  return 0;
}
