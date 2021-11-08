#include <chrono>
#include <iostream>
#include <cstdlib>
#include <cuda.h>
#include "kernels.h"

// thread block size
#define BLOCK_SIZE 256

template <typename T>
inline void memcpyH2D(T *d, const T *h, const int n) {
  cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void test (const int repeat, const int numFloats) 
{
  // Initialize host data, with the first half the same as the second
  T *hostMem = (T*) malloc (sizeof(T) * numFloats);

  srand48(123);
  for (int j = 0; j < numFloats/2 ; ++j)
    hostMem[j] = hostMem[numFloats-j-1] = (T)(drand48()*10.0);

  T *deviceMem;
  cudaMalloc((void**)&deviceMem, numFloats * sizeof(T));

  dim3 threads(BLOCK_SIZE);
  dim3 blocks((numFloats)/BLOCK_SIZE);

  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now(); 

  for (int i = 0; i < 100; i++) 
  {
    memcpyH2D(deviceMem, hostMem, numFloats);
    Add1<T><<< blocks, threads >>>(deviceMem, repeat, 10.0);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Add2<T><<< blocks, threads >>>(deviceMem, repeat, 10.0);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Add4<T><<< blocks, threads >>>(deviceMem, repeat, 10.0);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Add8<T><<< blocks, threads >>>(deviceMem, repeat, 10.0);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Mul1<T><<< blocks, threads >>>(deviceMem, repeat, 1.01);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Mul2<T><<< blocks, threads >>>(deviceMem, repeat, 1.01);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Mul4<T><<< blocks, threads >>>(deviceMem, repeat, 1.01);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    Mul8<T><<< blocks, threads >>>(deviceMem, repeat, 1.01);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MAdd1<T><<< blocks, threads >>>(deviceMem, repeat, 10.0, 0.9899);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MAdd2<T><<< blocks, threads >>>(deviceMem, repeat, 10.0, 0.9899);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MAdd4<T><<< blocks, threads >>>(deviceMem, repeat, 10.0, 0.9899);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MAdd8<T><<< blocks, threads >>>(deviceMem, repeat, 10.0, 0.9899);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MulMAdd1<T><<< blocks, threads >>>(deviceMem, repeat, 3.75, 0.355);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MulMAdd2<T><<< blocks, threads >>>(deviceMem, repeat, 3.75, 0.355);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MulMAdd4<T><<< blocks, threads >>>(deviceMem, repeat, 3.75, 0.355);
    cudaDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    MulMAdd8<T><<< blocks, threads >>>(deviceMem, repeat, 3.75, 0.355);
    cudaDeviceSynchronize();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << seconds << " s\n";
  
  cudaFree(deviceMem);
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
