// ************************************************
// original authors: Lee Howes and David B. Thomas
// ************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "loopback.h"
#include "kernels.cpp"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <dump> <repeat>\n", argv[0]);
    return 1;
  }
  // display device results when enabled
  const int dump = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t loopback_size = sizeof(float) * LOOKBACK_NUM_PARAMETER_VALUES;
  const size_t seed_size = sizeof(unsigned int) * TAUSWORTHE_NUM_SEEDS;

  float *lookback_VOL_0 = (float *) malloc(loopback_size);
  float *lookback_A_0 = (float *) malloc(loopback_size);
  float *lookback_A_1 = (float *) malloc(loopback_size);
  float *lookback_A_2 = (float *) malloc(loopback_size);
  float *lookback_S_0 = (float *) malloc(loopback_size);
  float *lookback_EPS_0 = (float *) malloc(loopback_size);
  float *lookback_MU = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsMean = (float *) malloc(loopback_size);
  float *lookbackSimulationResultsVariance = (float *) malloc(loopback_size);

  for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i++)
  {
    lookback_VOL_0[i] = Rand();
    lookback_A_0[i] = Rand();
    lookback_A_1[i] = Rand();
    lookback_A_2[i] = Rand();
    lookback_S_0[i] = Rand();
    lookback_EPS_0[i] = Rand();
    lookback_MU[i] = Rand();
  }

  unsigned int *tauswortheSeeds = (unsigned int *) malloc(seed_size);
  for (unsigned i = 0; i < TAUSWORTHE_NUM_SEEDS; i++)
    tauswortheSeeds[i] = (uint)rand() + 16;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned int *d_tauswortheSeeds = sycl::malloc_device<unsigned int>(TAUSWORTHE_NUM_SEEDS, q);
  float *d_lookback_VOL_0 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_A_0 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_A_1 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_A_2 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_S_0 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_EPS_0 = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookback_MU = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookbackSimulationResultsMean = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);
  float *d_lookbackSimulationResultsVariance = sycl::malloc_device<float>(LOOKBACK_NUM_PARAMETER_VALUES, q);

  q.memcpy(d_tauswortheSeeds, tauswortheSeeds, seed_size);
  q.memcpy(d_lookback_VOL_0, lookback_VOL_0, loopback_size);
  q.memcpy(d_lookback_A_0, lookback_A_0, loopback_size);
  q.memcpy(d_lookback_A_1, lookback_A_1, loopback_size);
  q.memcpy(d_lookback_A_2, lookback_A_2, loopback_size);
  q.memcpy(d_lookback_S_0, lookback_S_0, loopback_size);
  q.memcpy(d_lookback_EPS_0, lookback_EPS_0, loopback_size);
  q.memcpy(d_lookback_MU, lookback_MU, loopback_size);

  // Execute the Tausworthe version of the lookback option
  sycl::range<1> gws (LOOKBACK_TAUSWORTHE_NUM_BLOCKS * LOOKBACK_TAUSWORTHE_NUM_THREADS);
  sycl::range<1> lws (LOOKBACK_TAUSWORTHE_NUM_THREADS);
  const unsigned num_cycles = LOOKBACK_MAX_T;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> sm (sycl::range<1>(
        LOOKBACK_TAUSWORTHE_NUM_THREADS*LOOKBACK_MAX_T), cgh);
      cgh.parallel_for<class loopback>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        tausworthe_lookback(
          num_cycles,
          d_tauswortheSeeds,
          d_lookbackSimulationResultsMean,
          d_lookbackSimulationResultsVariance,
          d_lookback_VOL_0,
          d_lookback_EPS_0,
          d_lookback_A_0,
          d_lookback_A_1,
          d_lookback_A_2,
          d_lookback_S_0,
          d_lookback_MU,
          sm.get_pointer(),
	  item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  q.memcpy(lookbackSimulationResultsMean, d_lookbackSimulationResultsMean, loopback_size);
  q.memcpy(lookbackSimulationResultsVariance, d_lookbackSimulationResultsVariance, loopback_size);
  q.wait();

  if (dump) {
    for (unsigned i = 0; i < LOOKBACK_NUM_PARAMETER_VALUES; i++)
      printf("%d %.3f %.3f\n", i, lookbackSimulationResultsMean[i],
                              lookbackSimulationResultsVariance[i]);
  }

  free(lookback_VOL_0);
  free(lookback_A_0);
  free(lookback_A_1);
  free(lookback_A_2);
  free(lookback_S_0);
  free(lookback_EPS_0);
  free(lookback_MU);
  free(lookbackSimulationResultsMean);
  free(lookbackSimulationResultsVariance);
  free(tauswortheSeeds);

  sycl::free(d_tauswortheSeeds, q);
  sycl::free(d_lookback_VOL_0, q);
  sycl::free(d_lookback_A_0, q);
  sycl::free(d_lookback_A_1, q);
  sycl::free(d_lookback_A_2, q);
  sycl::free(d_lookback_S_0, q);
  sycl::free(d_lookback_EPS_0, q);
  sycl::free(d_lookback_MU, q);
  sycl::free(d_lookbackSimulationResultsMean, q);
  sycl::free(d_lookbackSimulationResultsVariance, q);
  return 0;
}
