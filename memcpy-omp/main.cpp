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

int main() {
  int *A;
  size_t size[NUM_SIZE];

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);

    size_t len = size[i] / sizeof(int);

    double uS;
    clock_t start, end;

    #pragma omp target data map(alloc: A[0:len])
    {
      start = clock();
      for (int j = 0; j < NUM_ITER; j++) {
        #pragma omp target update to (A[0:len]) nowait
      }
      end = clock();
      uS = (double)(end - start) * 1000 / (NUM_ITER * CLOCKS_PER_SEC);
    }

    std::cout << "Copy " << size[i] << " btyes from host to device takes " 
      << uS <<  " us" << std::endl;
    free(A);
  }
  return 0;
}
