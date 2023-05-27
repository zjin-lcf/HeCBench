/*
   Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <sycl/sycl.hpp>

#define NUM_SIZE 19  //size up to 16M
#define NUM_ITER 500 //Total GPU memory up to 16M*500=8G

#define Clock() std::chrono::steady_clock::now()

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

void setup(size_t *size, int &num, int **pA, const size_t totalGlobalMem) {

  for (int i = 0; i < num; i++) {
    size[i] = 1 << (i + 6);
    if((NUM_ITER + 1) * size[i] > totalGlobalMem) {
      num = i;
      break;
    }
  }
  *pA = (int*)malloc(size[num - 1]);
  valSet(*pA, 1, size[num - 1]);
}

void testInit(sycl::queue &q, size_t size, int type) {
  printf("Initial allocation and deallocation\n");

  int *Ad = nullptr;
  auto start = Clock();
  if (type == 0)
    Ad = (int *)sycl::malloc_shared(size, q);
  else if (type == 1)
    Ad = (int *)sycl::malloc_device(size, q);
  else if (type == 2)
    Ad = (int *)sycl::malloc_host(size, q);

  auto end = Clock();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  if (type == 0)
    printf("malloc_shared(%zu) takes %lf us\n", size, time * 1e-3);
  else if (type == 1)
    printf("malloc_device(%zu) takes %lf us\n", size, time * 1e-3);
  else if (type == 2)
    printf("malloc_host(%zu) takes %lf us\n", size, time * 1e-3);
  
  start = Clock();
  sycl::free(Ad, q);
  end = Clock();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("free(%zu) takes %lf us\n", size, time * 1e-3);

  printf("\n");
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <total global memory size in bytes>\n", argv[0]);
    return 1;
  }
   
  const size_t totalGlobalMem = atol(argv[1]);

  size_t size[NUM_SIZE] = { 0 };
  int *Ad[NUM_ITER] = { nullptr };

  int num = NUM_SIZE;
  int *A;
  setup(size, num, &A, totalGlobalMem);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("\n==== Evaluate malloc_shared and free ====\n");
  testInit(q, size[0], 0);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      Ad[j] = (int *)sycl::malloc_shared(size[i], q);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("malloc_shared(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      sycl::free(Ad[j], q);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("free(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  printf("\n==== Evaluate malloc_device and free ====\n");
  testInit(q, size[0], 1);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      Ad[j] = (int *)sycl::malloc_device(size[i], q);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("malloc_device(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      sycl::free(Ad[j], q);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("free(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  printf("\n==== Evaluate malloc_host and free ====\n");
  testInit(q, size[0], 2);

  for (int i = 0; i < num; i++) {
    auto start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      Ad[j] = (int *)sycl::malloc_host(size[i], q);
    }
    auto end = Clock();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("malloc_host(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);

    start = Clock();
    for (int j = 0; j < NUM_ITER; j++) {
      sycl::free(Ad[j], q);
      Ad[j] = nullptr;
    }
    end = Clock();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("free(%zu) takes %lf us\n", size[i], time * 1e-3  / NUM_ITER);
  }

  free(A);
  return 0;
}
