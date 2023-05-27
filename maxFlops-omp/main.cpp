#include <chrono>
#include <iostream>
#include <cstdlib>
#include <omp.h>
#include "kernels.h"

template <typename T>
void test (const int repeat, const int numFloats) 
{
  // Initialize host data, with the first half the same as the second
  T *hostMem = (T*) malloc (sizeof(T) * numFloats);

  srand48(123);
  for (int j = 0; j < numFloats/2 ; ++j)
    hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);

  #pragma omp target data map(alloc: hostMem[0:numFloats])
  {
    // warmup
    for (int i = 0; i < 4; i++) {
      Add1<T>(hostMem, numFloats, repeat, 10.0);
      Add2<T>(hostMem, numFloats, repeat, 10.0);
      Add4<T>(hostMem, numFloats, repeat, 10.0);
      Add8<T>(hostMem, numFloats, repeat, 10.0);
    }

    #pragma omp target update to (hostMem[0:numFloats])
    auto k_start = std::chrono::high_resolution_clock::now(); 
    Add1<T>(hostMem, numFloats, repeat, 10.0);
    auto k_end = std::chrono::high_resolution_clock::now(); 
    auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Add1): %f (s)\n", (k_time * 1e-9f));

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Add2<T>(hostMem, numFloats, repeat, 10.0);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Add2): %f (s)\n", (k_time * 1e-9f));

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Add4<T>(hostMem, numFloats, repeat, 10.0);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Add4): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Add8<T>(hostMem, numFloats, repeat, 10.0);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Add8): %f (s)\n", k_time * 1e-9f);

    // warmup
    for (int i = 0; i < 4; i++) {
      Mul1<T>(hostMem, numFloats, repeat, 1.01);
      Mul2<T>(hostMem, numFloats, repeat, 1.01);
      Mul4<T>(hostMem, numFloats, repeat, 1.01);
      Mul8<T>(hostMem, numFloats, repeat, 1.01);
    }

    k_start = std::chrono::high_resolution_clock::now(); 
    Mul1<T>(hostMem, numFloats, repeat, 1.01);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Mul1): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Mul2<T>(hostMem, numFloats, repeat, 1.01);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Mul2): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Mul4<T>(hostMem, numFloats, repeat, 1.01);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Mul4): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    Mul8<T>(hostMem, numFloats, repeat, 1.01);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (Mul8): %f (s)\n", k_time * 1e-9f);

    // warmup
    for (int i = 0; i < 4; i++) {
      MAdd1<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
      MAdd2<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
      MAdd4<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
      MAdd8<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
    }

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MAdd1<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MAdd1): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MAdd2<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MAdd2): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MAdd4<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MAdd4): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MAdd8<T>(hostMem, numFloats, repeat, 10.0, 0.9899);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MAdd8): %f (s)\n", k_time * 1e-9f);

    // warmup
    for (int i = 0; i < 4; i++) {
      MulMAdd1<T>(hostMem, numFloats, repeat, 3.75, 0.355);
      MulMAdd2<T>(hostMem, numFloats, repeat, 3.75, 0.355);
      MulMAdd4<T>(hostMem, numFloats, repeat, 3.75, 0.355);
      MulMAdd8<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    }

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MulMAdd1<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MulMAdd1): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MulMAdd2<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MulMAdd2): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MulMAdd4<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MulMAdd4): %f (s)\n", k_time * 1e-9f);

    #pragma omp target update to (hostMem[0:numFloats])
    k_start = std::chrono::high_resolution_clock::now(); 
    MulMAdd8<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    k_end = std::chrono::high_resolution_clock::now(); 
    k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
    printf("kernel execution time (MulMAdd8): %f (s)\n", k_time * 1e-9f);
  }
  
  free(hostMem);
}

int main(int argc, char* argv[]) 
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  // the number of loop iterations inside kernels
  const int repeat = atoi(argv[1]);

  // a multiple of BLOCK_SIZE
  const int numFloats = 2*1024*1024;

  printf("=== Single-precision floating-point kernels ===\n");
  test<float>(repeat, numFloats);

  // comment out when double-precision is not supported by a device
  printf("=== Double-precision floating-point kernels ===\n");
  test<double>(repeat, numFloats);

  return 0;
}
