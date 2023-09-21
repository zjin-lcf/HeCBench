#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>

#include "rand_helpers.h"
#include "constants.h"
#include "wallace_kernel.cu"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++) {
    float x = RandN();
    hostPool[i] = x;
  }

  float* rngChi2Corrections = (float *) malloc(4 * WALLACE_CHI2_COUNT);
  for (unsigned int i = 0; i < WALLACE_CHI2_COUNT; i++)
  {
    rngChi2Corrections[i] = MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE);
  }
  float* randomNumbers = (float *) malloc(4 * WALLACE_OUTPUT_SIZE);

  // device buffers
  float *device_randomNumbers;
  float *devicerngChi2Corrections;
  float *devPool;
  hipMalloc((void **) &devPool, 4 * WALLACE_TOTAL_POOL_SIZE);
  hipMalloc((void **) &devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT);
  hipMalloc((void **) &device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE);

  hipMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE, hipMemcpyHostToDevice);
  hipMemcpy(devicerngChi2Corrections, rngChi2Corrections,
             4 * WALLACE_CHI2_COUNT, hipMemcpyHostToDevice);

  dim3 rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
  dim3 rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);
  const unsigned seed = 1;

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(rng_wallace, rng_wallace_grid, rng_wallace_threads , 0, 0, 
        seed, devPool, device_randomNumbers, devicerngChi2Corrections);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  hipMemcpy(randomNumbers, device_randomNumbers,
             4 * WALLACE_OUTPUT_SIZE, hipMemcpyDeviceToHost);
#ifdef DEBUG
  // random numbers are different for each i iteration 
  for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
    printf("%.3f\n", randomNumbers[n]);
#endif

  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
  hipFree(devicerngChi2Corrections);
  hipFree(device_randomNumbers);
  hipFree(devPool);
  return 0;
}
