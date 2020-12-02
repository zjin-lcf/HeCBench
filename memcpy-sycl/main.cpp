#include <iostream>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include "common.h"

#define NUM_SIZE 16
#define NUM_ITER (1 << 13)

void setup(size_t *size) {
  for (int i = 0; i < NUM_SIZE; i++) {
    size[i] = 1 << (i + 6);  // start at 8 bytes
  }
}

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

int main() {
#ifdef USE_GPU 
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

  size_t size[NUM_SIZE];

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    int* A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);

    // create a device buffer to receive data from and send data to host
    size_t len = size[i] / sizeof(int);
    buffer<int, 1> d_A (len);

    clock_t start, end;
    double uS;

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_discard_write>(h);
        h.copy(A, d_A_acc);
      });
    }
    q.wait();
    end = clock();

    uS  = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_read>(h);
        h.copy(d_A_acc, A);
      });
    }
    q.wait();
    end = clock();
    double uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from device to host takes " 
      << uS <<  " us" << std::endl;

    free(A);
  }
  return 0;
}
