#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <omp.h>

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

  size_t size[NUM_SIZE];

  setup(size);
  for (int i = 0; i < NUM_SIZE; i++) {
    int* A = (int*)malloc(size[i]);
    if (A == nullptr) {
      std::cerr << "Host memory allocation failed\n";
      return -1;
    }	
    valSet(A, 1, size[i]);

    size_t len = size[i] / sizeof(int);

    #pragma omp target data map(alloc: A[0:len])
    {
      // warmup
      for (int j = 0; j < repeat; j++) {
        #pragma omp target update to (A[0:len])
      }

      auto start = std::chrono::steady_clock::now();

      for (int j = 0; j < repeat; j++) {
        #pragma omp target update to (A[0:len])
      }

      auto end = std::chrono::steady_clock::now();
      auto timeH2D = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::cout << "Copy " << size[i] << " bytes from host to device takes "
                << (timeH2D * 1e-3f) / repeat <<  " us" << std::endl;

      // warmup
      for (int j = 0; j < repeat; j++) {
        #pragma omp target update from (A[0:len])
      }

      start = std::chrono::steady_clock::now();

      for (int j = 0; j < repeat; j++) {
        #pragma omp target update from (A[0:len])
      }

      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      auto timeD2H = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      std::cout << "Copy " << size[i] << " bytes from device to host takes "
                << (timeD2H * 1e-3f) / repeat <<  " us" << std::endl;
    }
    free(A);
    std::cout << "Timing gap in nanoseconds per byte: "
              << (float)std::abs(timeH2D - timeD2H) / (repeat * size[i]);
    std::cout << std::endl << std::endl;
  }
  return 0;
}
