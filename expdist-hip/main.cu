#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <hip/hip_runtime.h>
#include "kernel.h"

template <typename FP>
void test(const int size) {
  const int max_blocks = (int)(ceilf(size * size / 256.f)); 

  std::default_random_engine rng (123);
  std::normal_distribution<FP> distribution(0, 1);

  FP *A = (FP*) malloc (sizeof(FP) * size * 2);
  FP *B = (FP*) malloc (sizeof(FP) * size * 2);
  for (int i = 0; i < size * 2; i++) {
    A[i] = distribution(rng);
    B[i] = A[i] + (FP)0.00001 * distribution(rng);
  }

  FP *scaleA = (FP*) malloc (sizeof(FP) * size);
  FP *scaleB = (FP*) malloc (sizeof(FP) * size);
  for (int i = 0; i < size; i++) {
    scaleA[i] = (FP)0.01 * distribution(rng);
    if (scaleA[i] < (FP)0.0) scaleA[i] = -scaleA[i];
    scaleB[i] = (FP)0.01 * distribution(rng);
    if (scaleB[i] < (FP)0.0) scaleB[i] = -scaleB[i];
  }

  FP *cost = (FP*) malloc (sizeof(FP) * max_blocks);
  FP output;

  FP *d_A;
  FP *d_B;
  FP *d_scaleA;
  FP *d_scaleB;
  FP *d_cost;
  FP *d_output;

  hipMalloc((void**)&d_A, sizeof(FP) * size * 2);
  hipMalloc((void**)&d_B, sizeof(FP) * size * 2);
  hipMalloc((void**)&d_scaleA, sizeof(FP) * size);
  hipMalloc((void**)&d_scaleB, sizeof(FP) * size);
  hipMalloc((void**)&d_cost, sizeof(FP) * max_blocks);
  hipMalloc((void**)&d_output, sizeof(FP));

  hipMemcpy(d_A, A, sizeof(FP) * size * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, sizeof(FP) * size * 2, hipMemcpyHostToDevice);
  hipMemcpy(d_scaleA, scaleA, sizeof(FP) * size, hipMemcpyHostToDevice);
  hipMemcpy(d_scaleB, scaleB, sizeof(FP) * size, hipMemcpyHostToDevice);

  dim3 grids (size / (block_size_x * tile_size_x), 
              size / (block_size_y * tile_size_y));
  dim3 blocks (block_size_x, block_size_y);

  const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                      ceilf(size / (block_size_y * tile_size_y));

  for (int i = 0; i < 100; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(distance<FP>), grids, blocks, 0, 0, d_A, d_B, size, size, d_scaleA, d_scaleB, d_cost);  
    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_cross_term<FP>), 1, reduce_block_size, 0, 0, d_output, d_cost, size, size, nblocks);  
  }

  hipMemcpy(&output, d_output, sizeof(FP), hipMemcpyDeviceToHost);
  printf("output value: %lf\n", output);

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_scaleA);
  hipFree(d_scaleB);
  hipFree(d_output);
  hipFree(d_cost);

  free(A);
  free(B);
  free(scaleA);
  free(scaleB);
  free(cost);
} 

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage ./%s <size>\n", argv[0]);
    return 1;
  }

  const int size = atoi(argv[1]);

  printf("Test single precision\n");
  test<float>(size);

  printf("Test double precision\n");
  test<double>(size);

  return 0;
}
