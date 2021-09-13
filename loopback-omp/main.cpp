// ************************************************
// original authors: Lee Howes and David B. Thomas
// ************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "loopback.h"
#include "kernels.cpp"

int main(int argc, char* argv[]) {

  // display device results when enabled
  int dump = 0;
  if (argc == 2) dump = atoi(argv[1]);

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

#pragma omp target data map(to:\
  tauswortheSeeds[0:TAUSWORTHE_NUM_SEEDS], \
  lookback_VOL_0[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_A_0[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_A_1[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_A_2[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_S_0[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_EPS_0[0:LOOKBACK_NUM_PARAMETER_VALUES], \
  lookback_MU[0:LOOKBACK_NUM_PARAMETER_VALUES]) \
  map(from: lookbackSimulationResultsMean[0:LOOKBACK_NUM_PARAMETER_VALUES], \
            lookbackSimulationResultsVariance[0:LOOKBACK_NUM_PARAMETER_VALUES])
  {

  // Execute the Tausworthe version of the lookback option
  const unsigned num_cycles = LOOKBACK_MAX_T;

  for (int i = 0; i < 100; i++)
    tausworthe_lookback (
       num_cycles,
       tauswortheSeeds,
       lookbackSimulationResultsMean,
       lookbackSimulationResultsVariance,
       lookback_VOL_0,
       lookback_EPS_0,
       lookback_A_0,
       lookback_A_1,
       lookback_A_2,
       lookback_S_0,
       lookback_MU);
  }

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
