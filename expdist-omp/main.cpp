#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <omp.h>
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
  FP *cost = (FP*) malloc (sizeof(FP) * max_blocks);

#pragma omp target data map(to: A[0:size*2], \
                                B[0:size*2], \
                                scaleA[0:size], \
                                scaleB[0:size]) \
                        map(alloc: cost[0:max_blocks])
  {
    const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                        ceilf(size / (block_size_y * tile_size_y));

    FP output;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      distance<FP>(A, B, size, size, scaleA, scaleB, cost);  
      output = reduce_cross_term<FP>(cost, size, size, nblocks);  
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
    printf("output value: %lf\n", output);
  }

  free(A);
  free(B);
  free(cost);
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
