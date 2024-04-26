/*
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <chrono>
#include <omp.h>


#define DO_NOT_OPTIMIZE_AWAY                                                       \
  unsigned i = omp_get_num_teams() * omp_get_num_threads() + omp_get_thread_num(); \
  if (out) *out = args.args[i];

#pragma omp declare target
struct SmallKernelArgs {
  char args[16];
};

struct MediumKernelArgs {
  char args[256];
};

struct LargeKernelArgs {
  char args[4096];
};

void KernelWithSmallArgs(SmallKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithMediumArgs(MediumKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithLargeArgs(LargeKernelArgs args, char* out) { DO_NOT_OPTIMIZE_AWAY; }
#pragma omp end declare target

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  SmallKernelArgs small_kernel_args;
  MediumKernelArgs medium_kernel_args;
  LargeKernelArgs large_kernel_args;

  // warmup
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:small_kernel_args)
    KernelWithSmallArgs(small_kernel_args, nullptr);
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:small_kernel_args)
    KernelWithSmallArgs(small_kernel_args, nullptr);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithSmallArgs: %f (us)\n", (time * 1e-3f) / repeat);

  // warmup
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:medium_kernel_args)
    KernelWithMediumArgs(medium_kernel_args, nullptr);
  }

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:medium_kernel_args)
    KernelWithMediumArgs(medium_kernel_args, nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithMediumArgs: %f (us)\n", (time * 1e-3f) / repeat);

  // warmup
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:large_kernel_args)
    KernelWithLargeArgs(large_kernel_args, nullptr);
  }

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    #pragma omp target map(to:large_kernel_args)
    KernelWithLargeArgs(large_kernel_args, nullptr);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithLargeArgs: %f (us)\n", (time * 1e-3f) / repeat);

  return 0;
}
