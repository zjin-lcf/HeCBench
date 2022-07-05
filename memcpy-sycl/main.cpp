#include <iostream>
#include <cstdlib>
#include <chrono>
#include "common.h"

#define NUM_SIZE 16

void setup(size_t *size) {
  for (int i = 0; i < NUM_SIZE; i++) {
    size[i] = 1 << (i + 6);
  }
}

void valSet(int* A, int val, size_t size) {
  size_t len = size / sizeof(int);
  for (size_t i = 0; i < len; i++) {
    A[i] = val;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);

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

    // warmup
    for (int j = 0; j < repeat; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_discard_write>(h);
        h.copy(A, d_A_acc);
      });
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int j = 0; j < repeat; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_discard_write>(h);
        h.copy(A, d_A_acc);
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
              << (time * 1e-3f) / repeat <<  " us" << std::endl;

    // warmup
    for (int j = 0; j < repeat; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_read>(h);
        h.copy(d_A_acc, A);
      });
    }
    q.wait();

    start = std::chrono::steady_clock::now();
    
    for (int j = 0; j < repeat; j++) {
      q.submit([&](handler &h) {
        auto d_A_acc = d_A.get_access<sycl_read>(h);
        h.copy(d_A_acc, A);
      });
    }
    q.wait();

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Copy " << size[i] << " btyes from device to host takes " 
              << (time * 1e-3f) / repeat <<  " us" << std::endl;

    free(A);
  }
  return 0;
}
