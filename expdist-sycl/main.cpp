#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "common.h"
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

  FP output;

  { // sycl scope
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<FP, 1> d_A (A, size*2);
  buffer<FP, 1> d_B (B, size*2);
  buffer<FP, 1> d_scaleA (scaleA, size);
  buffer<FP, 1> d_scaleB (scaleB, size);
  buffer<FP, 1> d_cost (max_blocks);
  buffer<FP, 1> d_output (&output, 1);

  range<2> gws (size / (block_size_y * tile_size_y) * block_size_y,
                size / (block_size_x * tile_size_x) * block_size_x);
  range<2> lws (block_size_y, block_size_x);

  range<1> gws2 (reduce_block_size);
  range<1> lws2 (reduce_block_size);

  const int nblocks = ceilf(size / (block_size_x * tile_size_x)) * 
                      ceilf(size / (block_size_y * tile_size_y));

  const int m = size;
  const int n = size;

  for (int i = 0; i < 100; i++) {
    q.submit([&] (handler &cgh) {
      auto A = d_A.template get_access<sycl_read>(cgh);
      auto B = d_B.template get_access<sycl_read>(cgh);
      auto sA = d_scaleA.template get_access<sycl_read>(cgh);
      auto sB = d_scaleB.template get_access<sycl_read>(cgh);
      auto c = d_cost.template get_access<sycl_discard_write>(cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_A (2 * block_size_x * tile_size_x, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_B (2 * block_size_y * tile_size_y, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_scaleA (block_size_x * tile_size_x, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sh_scaleB (block_size_y * tile_size_y, cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sum (1, cgh);
      cgh.parallel_for<class computeCost<FP>>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
        distance_tiled<FP>(
          item, A.get_pointer(), B.get_pointer(), m, n, 
          sA.get_pointer(), sB.get_pointer(), c.get_pointer(),
          sh_A.get_pointer(), sh_B.get_pointer(), 
          sh_scaleA.get_pointer(), sh_scaleB.get_pointer(), sum.get_pointer());
      });
    });

    q.submit([&] (handler &cgh) {
      auto o = d_output.template get_access<sycl_discard_write>(cgh);
      auto c = d_cost.template get_access<sycl_read>(cgh);
      accessor<FP, 1, sycl_read_write, access::target::local> sum (1, cgh);
      cgh.parallel_for<class reduceBlock<FP>>(nd_range<1>(gws2, lws2), [=] (nd_item<1> item) {
         reduce_cross_term<FP>(
           item, o.get_pointer(), c.get_pointer(), 
           sum.get_pointer(), m, n, nblocks);
      });
    });
  }

  } // sycl scope
  printf("output value: %lf\n", output);

  free(A);
  free(B);
  free(scaleA);
  free(scaleB);
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
