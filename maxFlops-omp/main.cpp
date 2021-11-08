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
    auto start = std::chrono::high_resolution_clock::now(); 

    for (int i = 0; i < 100; i++) 
    {
      #pragma omp target update to (hostMem[0:numFloats])
      Add1<T>(hostMem, numFloats, repeat, 10.0);

      #pragma omp target update to (hostMem[0:numFloats])
      Add2<T>(hostMem, numFloats, repeat, 10.0);

      #pragma omp target update to (hostMem[0:numFloats])
      Add4<T>(hostMem, numFloats, repeat, 10.0);

      #pragma omp target update to (hostMem[0:numFloats])
      Add8<T>(hostMem, numFloats, repeat, 10.0);

      #pragma omp target update to (hostMem[0:numFloats])
      Mul1<T>(hostMem, numFloats, repeat, 1.01);

      #pragma omp target update to (hostMem[0:numFloats])
      Mul2<T>(hostMem, numFloats, repeat, 1.01);

      #pragma omp target update to (hostMem[0:numFloats])
      Mul4<T>(hostMem, numFloats, repeat, 1.01);

      #pragma omp target update to (hostMem[0:numFloats])
      Mul8<T>(hostMem, numFloats, repeat, 1.01);

      #pragma omp target update to (hostMem[0:numFloats])
      MAdd1<T>(hostMem, numFloats, repeat, 10.0, 0.9899);

      #pragma omp target update to (hostMem[0:numFloats])
      MAdd2<T>(hostMem, numFloats, repeat, 10.0, 0.9899);

      #pragma omp target update to (hostMem[0:numFloats])
      MAdd4<T>(hostMem, numFloats, repeat, 10.0, 0.9899);

      #pragma omp target update to (hostMem[0:numFloats])
      MAdd8<T>(hostMem, numFloats, repeat, 10.0, 0.9899);

      #pragma omp target update to (hostMem[0:numFloats])
      MulMAdd1<T>(hostMem, numFloats, repeat, 3.75, 0.355);

      #pragma omp target update to (hostMem[0:numFloats])
      MulMAdd2<T>(hostMem, numFloats, repeat, 3.75, 0.355);

      #pragma omp target update to (hostMem[0:numFloats])
      MulMAdd4<T>(hostMem, numFloats, repeat, 3.75, 0.355);

      #pragma omp target update to (hostMem[0:numFloats])
      MulMAdd8<T>(hostMem, numFloats, repeat, 3.75, 0.355);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << seconds << " s\n";
  }
  
  free(hostMem);
}

int main(int argc, char* argv[]) 
{
  // the number of loop iterations inside kernels
  const int repeat = atoi(argv[1]);

  // a multiple of BLOCK_SIZE
  const int numFloats = 2*1024*1024;

  std::cout << "Total compute time of single-precision maxFLOPs: ";
  test<float>(repeat, numFloats);

  // comment out when double-precision is not supported by a device
  std::cout << "Total compute time of double-precision maxFLOPs: ";
  test<double>(repeat, numFloats);
}
