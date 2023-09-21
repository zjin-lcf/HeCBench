#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "kernel.h"

template <typename FP, int dim>
FP host_cost (FP *A, FP *B, FP *scale_A, FP *scale_B, int m, int n) {
  double sum = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      FP dist = 0;
      for (int d = 0; d < dim; d++) {
        dist += (A[i + d * m] - B[j + d * n]) *
                (A[i + d * m] - B[j + d * n]);
      }
      sum += exp(-dist/(scale_A[i] + scale_B[j]));
    }
  }
  return sum;
}

template <typename FP>
void test(const int size, const int repeat) {

  const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                      ceilf(size / (block_size_y * tile_size_y));

  size_t point_size_bytes = sizeof(FP) * size * 2;
  size_t scale_size_bytes = sizeof(FP) * size * 2;
  size_t cost_size_bytes = sizeof(FP) * nblocks;

  FP *A = (FP*) malloc (point_size_bytes);
  FP *B = (FP*) malloc (point_size_bytes);
  for (int i = 0; i < size * 2; i++) {
    A[i] = 1;
    B[i] = 0;
  }

  FP *scaleA = (FP*) malloc (scale_size_bytes);
  FP *scaleB = (FP*) malloc (scale_size_bytes);
  for (int i = 0; i < size; i++) {
    scaleA[i] = 1;
    scaleB[i] = 1;
  }

  FP *cost = (FP*) malloc (cost_size_bytes);

  #pragma omp target data map(to: A[0:size*2], \
                                  B[0:size*2], \
                                  scaleA[0:size], \
                                  scaleB[0:size]) \
                          map(alloc: cost[0:nblocks])
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

    printf("    device result: %lf\n", (double)output);

    output = host_cost<FP, 2>(A, B, scaleA, scaleB, size, size);
    printf("      host result: %lf\n", (double)output);

    printf("analytical result: %lf\n\n", size * size * exp(-1.0));
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
