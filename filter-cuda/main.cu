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
#include <chrono>
#include <cuda.h>

__global__ 
void filter(int *__restrict__ dst,
            int *__restrict__ nres,
            const int*__restrict__ src,
            int n)
{
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
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
    
  int *data_to_filter, *filtered_data, nres;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  data_to_filter = reinterpret_cast<int*>(malloc(sizeof(int) * num_elems));

  // Generate input data.
  srand(2);
  for (int i = 0; i < num_elems; i++) {
    data_to_filter[i] = rand() % 20;
  }

  cudaMalloc(&d_data_to_filter, sizeof(int) * num_elems);
  cudaMalloc(&d_filtered_data, sizeof(int) * num_elems);
  cudaMalloc(&d_nres, sizeof(int));

  cudaMemcpy(d_data_to_filter, data_to_filter,
             sizeof(int) * num_elems, cudaMemcpyHostToDevice);

  dim3 dimBlock (block_size);
  dim3 dimGrid ((num_elems + block_size - 1) / block_size);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_nres, 0, sizeof(int));
    filter<<<dimGrid, dimBlock>>>(d_filtered_data, d_nres, d_data_to_filter, num_elems);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(&nres, d_nres, sizeof(int), cudaMemcpyDeviceToHost);

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  cudaMemcpy(filtered_data, d_filtered_data, sizeof(int) * nres, cudaMemcpyDeviceToHost);

  int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres ? "PASS" : "FAIL");

  cudaFree(d_data_to_filter);
  cudaFree(d_filtered_data);
  cudaFree(d_nres);
  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);

  return 0;
}
