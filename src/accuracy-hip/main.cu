#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

__global__
void accuracy_kernel(
    const int N,
    const int D,
    const int top_k,
    const float* Xdata,
    const int* labelData,
    int* accuracy)
{
  typedef hipcub::BlockReduce<int, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int count = 0;

  for (int row = blockIdx.x; row < N; row += gridDim.x) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = threadIdx.x; col < D; col += blockDim.x) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    ngt = BlockReduce(temp_storage).Sum(ngt);
    if (ngt <= top_k) {
      ++count;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { 
    atomicAdd(accuracy, count);
  }
}
 
int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int); 
  const size_t data_size_bytes = data_size * sizeof(float); 

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims; 

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

  int *d_label;
  hipMalloc((void**)&d_label, label_size_bytes);
  hipMemcpy(d_label, label, label_size_bytes, hipMemcpyHostToDevice);

  float *d_data;
  hipMalloc((void**)&d_data, data_size_bytes);
  hipMemcpy(d_data, data, data_size_bytes, hipMemcpyHostToDevice);

  int *d_count;
  hipMalloc((void**)&d_count, sizeof(int));

  hipDeviceSynchronize();
  dim3 block (GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    dim3 grid (ngrid);
    printf("Grid size is %d\n", ngrid);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      hipMemset(d_count, 0, sizeof(int));
      accuracy_kernel<<<grid, block>>>(nrows, ndims, top_k, d_data, d_label, d_count);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    hipMemcpy(&count, d_count, sizeof(int), hipMemcpyDeviceToHost);
    bool ok = (count == count_ref);
    printf("%s\n", ok ? "PASS" : "FAIL");
    // printf("Accuracy = %f\n", (float)count / nrows);
  }

  hipFree(d_label);
  hipFree(d_data);
  hipFree(d_count);

  free(label);
  free(data);

  return 0;
}
