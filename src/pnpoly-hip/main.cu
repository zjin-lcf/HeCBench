#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <hip/hip_runtime.h>

#define VERTICES 600
#define BLOCK_SIZE_X 256

#include "kernel.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = VERTICES;

  std::default_random_engine rng (123);
  std::normal_distribution<float> distribution(0, 1);

  float2 *point = (float2*) malloc (sizeof(float2) * nPoints);
  for (int i = 0; i < nPoints; i++) {
    point[i].x = distribution(rng);
    point[i].y = distribution(rng);
  }

  float2 *vertex = (float2*) malloc (vertices * sizeof(float2));
  for (int i = 0; i < vertices; i++) {
    float t = distribution(rng) * 2.f * M_PI;
    vertex[i].x = cosf(t);
    vertex[i].y = sinf(t);
  }

  // kernel results
  int *bitmap_ref = (int*) malloc (nPoints * sizeof(int));
  int *bitmap_opt = (int*) malloc (nPoints * sizeof(int));

  float2 *d_point;
  float2 *d_vertex;
  int *d_bitmap_ref, *d_bitmap_opt;

  hipMalloc(&d_point, nPoints*sizeof(float2));
  hipMalloc(&d_vertex, vertices*sizeof(float2));
  hipMalloc(&d_bitmap_ref, nPoints*sizeof(int));
  hipMalloc(&d_bitmap_opt, nPoints*sizeof(int));

  //kernel parameters
  dim3 threads (BLOCK_SIZE_X);
  dim3 grid ((nPoints+BLOCK_SIZE_X-1)/BLOCK_SIZE_X);

  hipMemcpy(d_point, point, nPoints*sizeof(float2), hipMemcpyHostToDevice);
  hipMemcpy(d_vertex, vertex, vertices*sizeof(float2), hipMemcpyHostToDevice);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(pnpoly_base, grid, threads, 0, 0, d_bitmap_ref, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_base): %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(bitmap_ref, d_bitmap_ref, nPoints*sizeof(int), hipMemcpyDeviceToHost);

  // performance tuning with tile sizes

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<1>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<2>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<4>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<8>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<16>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<32>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n", (time * 1e-9f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(pnpoly_opt<64>), grid, threads, 0, 0, d_bitmap_opt, d_point, d_vertex, nPoints);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n", (time * 1e-9f) / repeat);

  hipMemcpy(bitmap_opt, d_bitmap_opt, nPoints*sizeof(int), hipMemcpyDeviceToHost);

  // compare against reference kernel for verification
  int error = memcmp(bitmap_opt, bitmap_ref, nPoints*sizeof(int)); 
  
  // double check
  int checksum = 0;
  for (int i = 0; i < nPoints; i++) checksum += bitmap_opt[i];
  printf("Checksum: %d\n", checksum);

  printf("%s\n", error ? "FAIL" : "PASS");

  hipFree(d_vertex);
  hipFree(d_point);
  hipFree(d_bitmap_ref);
  hipFree(d_bitmap_opt);

  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return error ? 1 : 0;
}
