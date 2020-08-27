#include <iostream>
#include <cstdlib>
#include <time.h>
#include <cuda.h>

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

int main() {
  int *A, *Ad;
  size_t size[NUM_SIZE];
  cudaError_t err;

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);
    err = cudaMalloc((void**)&Ad, size[i]);
    if (err != cudaSuccess) {
      std::cerr << "Device memory allocation failed\n";
      free(A);
      return -1;
    }
    clock_t start, end;
    start = clock();
    for (int j = 0; j < NUM_ITER; j++) {
      cudaMemcpy(Ad, A, size[i], cudaMemcpyHostToDevice);
    }
    //cudaDeviceSynchronize();
    end = clock();
    double uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;
    cudaFree(Ad);
    free(A);
  }
  return 0;
}
