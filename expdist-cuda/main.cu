#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <cuda.h>
#include "kernel.h"

template <typename FP>
void test() {
  const int alloc_size = 3000;
  const int size = 2000;
  const int max_blocks = (int)(ceilf(size * size / 256.f)); 

  std::default_random_engine rng (123);
  std::normal_distribution<FP> distribution(0, 1);

  FP *A = (FP*) malloc (sizeof(FP) * alloc_size * 2);
  FP *B = (FP*) malloc (sizeof(FP) * alloc_size * 2);
  for (int i = 0; i < alloc_size * 2; i++) {
    A[i] = distribution(rng);
    B[i] = A[i] + (FP)0.00001 * distribution(rng);
  }

  FP *scaleA = (FP*) malloc (sizeof(FP) * alloc_size);
  FP *scaleB = (FP*) malloc (sizeof(FP) * alloc_size);
  for (int i = 0; i < alloc_size; i++) {
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

  cudaMalloc((void**)&d_A, sizeof(FP) * alloc_size * 2);
  cudaMalloc((void**)&d_B, sizeof(FP) * alloc_size * 2);
  cudaMalloc((void**)&d_scaleA, sizeof(FP) * alloc_size);
  cudaMalloc((void**)&d_scaleB, sizeof(FP) * alloc_size);
  cudaMalloc((void**)&d_cost, sizeof(FP) * max_blocks);
  cudaMalloc((void**)&d_output, sizeof(FP));

  cudaMemcpy(d_A, A, sizeof(FP) * alloc_size * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(FP) * alloc_size * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scaleA, scaleA, sizeof(FP) * alloc_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scaleB, scaleB, sizeof(FP) * alloc_size, cudaMemcpyHostToDevice);

  dim3 grids (size / (block_size_x * tile_size_x), 
              size / (block_size_y * tile_size_y));
  dim3 blocks (block_size_x, block_size_y);

  const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                      ceilf(size / (block_size_y * tile_size_y));

  for (int i = 0; i < 100; i++) {
    distance<FP><<<grids, blocks>>>(d_A, d_B, size, size, d_scaleA, d_scaleB, d_cost);  
    reduce_cross_term<FP><<<1, 256>>>(d_output, d_cost, size, size, nblocks);  
  }

  cudaMemcpy(&output, d_output, sizeof(FP), cudaMemcpyDeviceToHost);
  printf("output value: %lf\n", output);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_scaleA);
  cudaFree(d_scaleB);
  cudaFree(d_output);
  cudaFree(d_cost);

  free(A);
  free(B);
  free(scaleA);
  free(scaleB);
  free(cost);
} 

int main() {
  printf("Test single precision\n");
  test<float>();
  printf("Test double precision\n");
  test<double>();
  return 0;
}
