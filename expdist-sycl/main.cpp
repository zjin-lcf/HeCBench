#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include "common.h"
#include "kernel.h"

template <typename FP, int dim>
FP cost (FP *A, FP *B, FP *scale_A, FP *scale_B, int m, int n) {
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
  size_t scale_size_bytes = sizeof(FP) * size;
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
  FP output;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  FP *d_A = malloc_device<FP>(size*2, q);
  q.memcpy(d_A, A, point_size_bytes);

  FP *d_B = malloc_device<FP>(size*2, q);
  q.memcpy(d_B, B, point_size_bytes);

  FP *d_scaleA = malloc_device<FP>(size, q);
  q.memcpy(d_scaleA, scaleA, scale_size_bytes);

  FP *d_scaleB = malloc_device<FP>(size, q);
  q.memcpy(d_scaleB, scaleB, scale_size_bytes);

  FP *d_cost = malloc_device<FP>(nblocks, q);
  FP *d_output = malloc_device<FP>(1, q);

  range<2> gws (size / (block_size_y * tile_size_y) * block_size_y,
                size / (block_size_x * tile_size_x) * block_size_x);
  range<2> lws (block_size_y, block_size_x);

  range<1> gws2 (reduce_block_size);
  range<1> lws2 (reduce_block_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      accessor<FP, 1, sycl_read_write, access::target::local> sh_A (2 * block_size_x * tile_size_x, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_B (2 * block_size_y * tile_size_y, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_scaleA (block_size_x * tile_size_x, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_scaleB (block_size_y * tile_size_y, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sum (1, cgh);
      cgh.parallel_for<class computeCost<FP>>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        distance_tiled<FP>(
          item, d_A, d_B, size, size, d_scaleA, d_scaleB, d_cost,
          sh_A.get_pointer(), sh_B.get_pointer(), 
          sh_scaleA.get_pointer(), sh_scaleB.get_pointer(), sum.get_pointer());
      });
    });

    q.submit([&] (handler &cgh) {
      accessor<FP, 1, sycl_read_write, access::target::local> sum (1, cgh);
      cgh.parallel_for<class reduceBlock<FP>>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
         reduce_cross_term<FP>(
           item, d_output, d_cost, sum.get_pointer(), size, size, nblocks);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(&output, d_output, sizeof(FP)).wait();
  printf("    device result: %lf\n", (double)output);

  output = cost<FP, 2>(A, B, scaleA, scaleB, size, size);
  printf("      host result: %lf\n", (double)output);

  printf("analytical result: %lf\n\n", size * size * exp(-1.0));

  free(d_A, q);
  free(d_B, q);
  free(d_scaleA, q);
  free(d_scaleB, q);
  free(d_output, q);
  free(d_cost, q);
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
