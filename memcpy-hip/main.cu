#include <iostream>
#include <cstdlib>
#include <cstring>
#include <time.h>
#include <hip/hip_runtime.h>

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
  int *d_A;
  size_t size[NUM_SIZE];
  hipError_t err;

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    int* A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);

    err = hipMalloc((void**)&d_A, size[i]);
    if (err != hipSuccess) {
      std::cerr << "Device memory allocation failed\n";
      free(A);
      return -1;
    }

    clock_t start, end;
    double uS;

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      hipMemcpyAsync(d_A, A, size[i], hipMemcpyHostToDevice, 0);
    }
    hipDeviceSynchronize();
    end = clock();
    uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;

    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      hipMemcpyAsync(A, d_A, size[i], hipMemcpyDeviceToHost, 0);
    }
    hipDeviceSynchronize();
    end = clock();
    uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from device to host takes " 
      << uS <<  " us" << std::endl;

    hipFree(d_A);
    free(A);
  }
  return 0;
}
