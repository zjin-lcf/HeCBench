#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "rand_helpers.h"
#include "constants.h"
#include "wallace_kernel.cu"

int main() 
{
  // host buffers
  float *hostPool = (float *) malloc(4 * WALLACE_TOTAL_POOL_SIZE);
  for (unsigned i = 0; i < WALLACE_TOTAL_POOL_SIZE; i++)
  {
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
  cudaMalloc((void **) &devPool, 4 * WALLACE_TOTAL_POOL_SIZE);
  cudaMalloc((void **) &devicerngChi2Corrections, 4 * WALLACE_CHI2_COUNT);
  cudaMalloc((void **) &device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE);

  cudaMemcpy(devPool, hostPool, 4 * WALLACE_TOTAL_POOL_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(devicerngChi2Corrections, rngChi2Corrections, 4 * WALLACE_CHI2_COUNT, cudaMemcpyHostToDevice);
  
  dim3 rng_wallace_grid(WALLACE_NUM_BLOCKS, 1, 1);
  dim3 rng_wallace_threads(WALLACE_NUM_THREADS, 1, 1);
  const unsigned seed = 1;
  
  for (int i = 0; i < 100; i++) {
    rng_wallace <<< rng_wallace_grid, rng_wallace_threads >>> (
      seed, devPool, device_randomNumbers, devicerngChi2Corrections);
    cudaMemcpy(randomNumbers, device_randomNumbers, 4 * WALLACE_OUTPUT_SIZE, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    // random numbers are different for each i iteration 
    for (unsigned int n = 0; n < WALLACE_OUTPUT_SIZE; n++) 
    	printf("%.3f\n", randomNumbers[n]);
#endif
  }
  
  free(rngChi2Corrections);
  free(randomNumbers);
  free(hostPool);
  cudaFree(devicerngChi2Corrections);
  cudaFree(device_randomNumbers);
  cudaFree(devPool);
  return 0;
}
