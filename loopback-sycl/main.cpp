// ************************************************
// original authors: Lee Howes and David B. Thomas
// ************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "common.h"
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

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned int,1> d_tauswortheSeeds (tauswortheSeeds, TAUSWORTHE_NUM_SEEDS);
  buffer<float,1> d_lookback_VOL_0 (lookback_VOL_0, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_A_0 (lookback_A_0, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_A_1 (lookback_A_1, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_A_2 (lookback_A_2, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_S_0 (lookback_S_0, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_EPS_0 (lookback_EPS_0, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookback_MU (lookback_MU, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookbackSimulationResultsMean (lookbackSimulationResultsMean, LOOKBACK_NUM_PARAMETER_VALUES);
  buffer<float,1> d_lookbackSimulationResultsVariance (lookbackSimulationResultsVariance, LOOKBACK_NUM_PARAMETER_VALUES);

  // Execute the Tausworthe version of the lookback option
  range<1> gws (LOOKBACK_TAUSWORTHE_NUM_BLOCKS * LOOKBACK_TAUSWORTHE_NUM_THREADS); 
  range<1> lws (LOOKBACK_TAUSWORTHE_NUM_THREADS);
  const unsigned num_cycles = LOOKBACK_MAX_T;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (handler &cgh) {
      auto mean = d_lookbackSimulationResultsMean.get_access<sycl_discard_write>(cgh);
      auto var  = d_lookbackSimulationResultsVariance.get_access<sycl_discard_write>(cgh);
      auto seed = d_tauswortheSeeds.get_access<sycl_read>(cgh);
      auto vol  = d_lookback_VOL_0.get_access<sycl_read>(cgh);
      auto eps  = d_lookback_EPS_0.get_access<sycl_read>(cgh);
      auto a0   = d_lookback_A_0.get_access<sycl_read>(cgh);
      auto a1   = d_lookback_A_1.get_access<sycl_read>(cgh);
      auto a2   = d_lookback_A_2.get_access<sycl_read>(cgh);
      auto s0   = d_lookback_S_0.get_access<sycl_read>(cgh);
      auto mu   = d_lookback_MU.get_access<sycl_read>(cgh);
      accessor<float, 1, sycl_read_write, access::target::local> sm(
        LOOKBACK_TAUSWORTHE_NUM_THREADS*LOOKBACK_MAX_T, cgh);
      cgh.parallel_for<class loopback>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        tausworthe_lookback(
          num_cycles,
          seed.get_pointer(),
          mean.get_pointer(),
          var.get_pointer(),
          vol.get_pointer(),
          eps.get_pointer(),
          a0.get_pointer(),
          a1.get_pointer(),
          a2.get_pointer(),
          s0.get_pointer(),
          mu.get_pointer(),
          sm.get_pointer(),
	  item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
  
  } // sycl scope

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

  return 0;
}
