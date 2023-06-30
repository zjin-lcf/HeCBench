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

  // warmup
  for (int i = 0; i < 4; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add1<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add2<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add4<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Add8<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
    hipDeviceSynchronize();
  }

  memcpyH2D(deviceMem, hostMem, numFloats);
  auto k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Add1<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
  hipDeviceSynchronize();
  auto k_end = std::chrono::high_resolution_clock::now(); 
  auto k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add1): %f (s)\n", (k_time * 1e-9f));

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Add2<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add2): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Add4<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add4): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Add8<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Add8): %f (s)\n", k_time * 1e-9f);

  // warmup
  for (int i = 0; i < 4; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul1<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul2<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul4<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul8<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
    hipDeviceSynchronize();
  }

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul1<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul1): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul2<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul2): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul4<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul4): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Mul8<T>), blocks, threads , 0, 0, deviceMem, repeat, 1.01);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (Mul8): %f (s)\n", k_time * 1e-9f);

  // warmup
  for (int i = 0; i < 4; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd1<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd2<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd4<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd8<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
    hipDeviceSynchronize();
  }

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd1<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd1): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd2<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd2): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd4<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd4): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MAdd8<T>), blocks, threads , 0, 0, deviceMem, repeat, 10.0, 0.9899);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MAdd8): %f (s)\n", k_time * 1e-9f);

  // warmup
  for (int i = 0; i < 4; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd1<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd2<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd4<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd8<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
    hipDeviceSynchronize();
  }

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd1<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd1): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd2<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd2): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd4<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd4): %f (s)\n", k_time * 1e-9f);

  memcpyH2D(deviceMem, hostMem, numFloats);
  k_start = std::chrono::high_resolution_clock::now(); 
  hipLaunchKernelGGL(HIP_KERNEL_NAME(MulMAdd8<T>), blocks, threads , 0, 0, deviceMem, repeat, 3.75, 0.355);
  hipDeviceSynchronize();
  k_end = std::chrono::high_resolution_clock::now(); 
  k_time = std::chrono::duration_cast<std::chrono::nanoseconds>(k_end - k_start).count();
  printf("kernel execution time (MulMAdd8): %f (s)\n", k_time * 1e-9f);

  hipFree(deviceMem);
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
