#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "kernel.h"

template <typename FP>
void test(const int size, const int repeat) {
  const int max_blocks = (int)(ceilf(size * size / 256.f)); 

  std::default_random_engine rng (123);
  std::normal_distribution<FP> distribution(0, 1);

  FP *A = (FP*) malloc (sizeof(FP) * size * 2);
  FP *B = (FP*) malloc (sizeof(FP) * size * 2);
  for (int i = 0; i < size * 2; i++) {
    A[i] = distribution(rng);
    B[i] = A[i] + distribution(rng);
  }

  FP *scaleA = (FP*) malloc (sizeof(FP) * size);
  FP *scaleB = (FP*) malloc (sizeof(FP) * size);
  for (int i = 0; i < size; i++) {
    scaleA[i] = (FP)0.01 * distribution(rng);
    if (scaleA[i] < (FP)0.0) scaleA[i] = -scaleA[i];
    scaleB[i] = (FP)0.01 * distribution(rng);
    if (scaleB[i] < (FP)0.0) scaleB[i] = -scaleB[i];
  }

  FP output;

  FP *d_A;
  FP *d_B;
  FP *d_scaleA;
  FP *d_scaleB;
  FP *d_cost;
  FP *d_output;

  cudaMalloc((void**)&d_A, sizeof(FP) * size * 2);
  cudaMalloc((void**)&d_B, sizeof(FP) * size * 2);
  cudaMalloc((void**)&d_scaleA, sizeof(FP) * size);
  cudaMalloc((void**)&d_scaleB, sizeof(FP) * size);
  cudaMalloc((void**)&d_cost, sizeof(FP) * max_blocks);
  cudaMalloc((void**)&d_output, sizeof(FP));

  cudaMemcpy(d_A, A, sizeof(FP) * size * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(FP) * size * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scaleA, scaleA, sizeof(FP) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scaleB, scaleB, sizeof(FP) * size, cudaMemcpyHostToDevice);

  dim3 grids (size / (block_size_x * tile_size_x), 
              size / (block_size_y * tile_size_y));
  dim3 blocks (block_size_x, block_size_y);

  const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                      ceilf(size / (block_size_y * tile_size_y));

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    distance<FP><<<grids, blocks>>>(d_A, d_B, size, size, d_scaleA, d_scaleB, d_cost);  
    reduce_cross_term<FP><<<1, reduce_block_size>>>(d_output, d_cost, size, size, nblocks);  
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

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
} 

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage ./%s <size> <repeat>\n", argv[0]);
    return 1;
  }

  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("Test single precision\n");
  test<float>(size, repeat);

  printf("Test double precision\n");
  test<double>(size, repeat);

  return 0;
}
