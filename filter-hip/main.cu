/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define NUM_ELEMS 100000000
#define NUM_THREADS_PER_BLOCK 256

__global__ 
void filter(int *dst, int *nres, const int* src, int n) {
  __shared__ int l_n;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // zero the counter
  if (threadIdx.x == 0)
    l_n = 0;
  __syncthreads();

  // get the value, evaluate the predicate, and
  // increment the counter if needed
  int d, pos;

  if(i < n) {
    d = src[i];
    if(d > 0)
      pos = atomicAdd(&l_n, 1);
  }
  __syncthreads();

  // leader increments the global counter
  if(threadIdx.x == 0)
    l_n = atomicAdd(nres, l_n);
  __syncthreads();

  // threads with true predicates write their elements
  if(i < n && d > 0) {
    pos += l_n; // increment local pos by global counter
    dst[pos] = d;
  }
  __syncthreads();
}


int main(int argc, char **argv) {
  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  srand(2);
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % 20;
  }

  hipMalloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS);
  hipMalloc(&d_filtered_data, sizeof(int) * NUM_ELEMS);
  hipMalloc(&d_nres, sizeof(int));

  hipMemcpy(d_data_to_filter, data_to_filter,
             sizeof(int) * NUM_ELEMS, hipMemcpyHostToDevice);
  hipMemcpy(d_nres, &nres, sizeof(int), hipMemcpyHostToDevice);

  dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  dim3 dimGrid( (NUM_ELEMS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK , 1, 1);

  hipLaunchKernelGGL(filter, dimGrid, dimBlock, 0, 0, d_filtered_data, d_nres, d_data_to_filter, NUM_ELEMS);

  hipMemcpy(&nres, d_nres, sizeof(int), hipMemcpyDeviceToHost);

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  hipMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres, hipMemcpyDeviceToHost);

  int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres ? "PASSED" : "FAILED");

  hipFree(d_data_to_filter);
  hipFree(d_filtered_data);
  hipFree(d_nres);
  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
