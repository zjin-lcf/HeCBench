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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cstdlib>
#include <time.h>

#define NUM_SIZE 16
#define NUM_ITER (1 << 13)

void setup(size_t *size) {
  for (int i = 0; i < NUM_SIZE; i++) {
    size[i] = 1 << (i + 6);  // start at 8 bytes
  }
}

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (int i = 0; i < len; i++) {
    A[i] = val;
  }
}

int main() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *A, *Ad;
  size_t size[NUM_SIZE];
  int err;

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);
    /*
    DPCT1003:2: Migrated API does not return error code. (*, 0) is inserted. You
    may need to rewrite this code.
    */
    err = (Ad = (int *)sycl::malloc_device(size[i], q_ct1), 0);
    /*
    DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
      std::cerr << "Device memory allocation failed\n";
      /*
      DPCT1001:0: The statement could not be removed.
      */
      free(A);
      return -1;
    }
    clock_t start, end;
    /*
    DPCT1008:3: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      q_ct1.memcpy(Ad, A, size[i]).wait();
    }
    //cudaDeviceSynchronize();
    /*
    DPCT1008:4: clock function is not defined in the DPC++. This is a
    hardware-specific feature. Consult with your hardware vendor to find a
    replacement.
    */
    end = clock();
    double uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;
    sycl::free(Ad, q_ct1);
    free(A);
  }
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
