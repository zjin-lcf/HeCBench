#include <chrono>
#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "kernels.h"

// thread block size
#define BLOCK_SIZE 256

template <typename T>
inline void memcpyH2D(T *d, const T *h, const int n) {
  hipMemcpy(d, h, n * sizeof(T), hipMemcpyHostToDevice);
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
  hipMalloc((void**)&deviceMem, numFloats * sizeof(T));

  dim3 threads(BLOCK_SIZE);
  dim3 blocks((numFloats)/BLOCK_SIZE);

  hipDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now(); 

  for (int i = 0; i < 100; i++) 
  {
    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add1<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add2<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add4<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add8<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul1<T>), blocks, threads, 0, 0, deviceMem, repeat, 1.01);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul2<T>), blocks, threads, 0, 0, deviceMem, repeat, 1.01);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul4<T>), blocks, threads, 0, 0, deviceMem, repeat, 1.01);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul8<T>), blocks, threads, 0, 0, deviceMem, repeat, 1.01);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd1<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd2<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd4<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd8<T>), blocks, threads, 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd1<T>), blocks, threads, 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd2<T>), blocks, threads, 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd4<T>), blocks, threads, 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipDeviceSynchronize();

    memcpyH2D(deviceMem, hostMem, numFloats);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd8<T>), blocks, threads, 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipDeviceSynchronize();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
  std::cout << seconds << " s\n";
  
  hipFree(deviceMem);
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
