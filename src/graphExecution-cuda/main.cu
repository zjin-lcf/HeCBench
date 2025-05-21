/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 256
#define LAUNCH_ITERATIONS  3


__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize, size_t outputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x*blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i=globaltid; i < inputSize; i+=gridDim.x*blockDim.x)
  {
    temp_sum += (double) inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  double beta  = temp_sum;
  double temp;

  for (int i = tile32.size() / 2; i > 0; i >>= 1) {
    if (tile32.thread_rank() < i) {
      temp       = tmp[cta.thread_rank() + i];
      beta       += temp;
      tmp[cta.thread_rank()] = beta;
    }
    cg::sync(tile32);
  }
  cg::sync(cta);

  if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
    beta  = 0.0;
    for (int i = 0; i < cta.size(); i += tile32.size()) {
      beta  += tmp[i];
    }
    outputVec[blockIdx.x] =  beta;
  }
}

__global__ void reduceFinal(double *inputVec, double *result, size_t inputSize)
{
  __shared__ double tmp[THREADS_PER_BLOCK];

  cg::thread_block cta = cg::this_thread_block();
  size_t globaltid = blockIdx.x*blockDim.x + threadIdx.x;

  double temp_sum = 0.0;
  for (int i=globaltid; i < inputSize; i+=gridDim.x*blockDim.x)
  {
    temp_sum += (double) inputVec[i];
  }
  tmp[cta.thread_rank()] = temp_sum;

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  // do reduction in shared mem
  if ((blockDim.x >= 512) && (cta.thread_rank() < 256))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
  }

  cg::sync(cta);

  if ((blockDim.x >= 256) &&(cta.thread_rank() < 128))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
  }

  cg::sync(cta);

  if ((blockDim.x >= 128) && (cta.thread_rank() <  64))
  {
    tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() +  64];
  }

  cg::sync(cta);

  if (cta.thread_rank() < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (blockDim.x >=  64) temp_sum += tmp[cta.thread_rank() + 32];
    // Reduce final warp using shuffle
    for (int offset = tile32.size()/2; offset > 0; offset /= 2) 
    {
      temp_sum += tile32.shfl_down(temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float*a, size_t size)
{
  srand(123);
  for (size_t i=0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void usingGraph(float* inputVec_h, float *inputVec_d,
    double *outputVec_d, double *result_d,
    size_t inputSize, size_t numOfBlocks)
{
  cudaStream_t stream;
  cudaGraph_t graph;

  cudaStreamCreate(&stream);

  cudaMemsetAsync(outputVec_d, 0, sizeof(double)*numOfBlocks, stream);
  cudaMemsetAsync(result_d, 0, sizeof(double), stream);

  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float)*inputSize, cudaMemcpyDefault, stream);

  for (int i = 0; i < 100; i++) {
    reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);
    reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream>>>(outputVec_d, result_d, numOfBlocks);
  }

  double result_h = 0.0;
  cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream);

  cudaStreamEndCapture(stream, &graph);

  cudaGraphExec_t graphExec;
  cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

  for (int i=0; i < LAUNCH_ITERATIONS; i++)
  {
    auto start = std::chrono::steady_clock::now();
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("[usingGraph] final reduced sum = %lf\n", result_h);
    printf("Execution time: %f (us)\n\n", (time * 1e-3f));
  }

  cudaGraphExecDestroy(graphExec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
}

void usingStream(float* inputVec_h, float *inputVec_d,
    double *outputVec_d, double *result_d,
    size_t inputSize, size_t numOfBlocks)
{
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  cudaMemsetAsync(outputVec_d, 0, sizeof(double)*numOfBlocks, stream);
  cudaMemsetAsync(result_d, 0, sizeof(double), stream);

  for (int i=0; i < LAUNCH_ITERATIONS; i++) {

    auto start = std::chrono::steady_clock::now();

    cudaMemcpyAsync(inputVec_d, inputVec_h, sizeof(float)*inputSize, cudaMemcpyDefault, stream);

    for (int i = 0; i < 100; i++) {
      reduce<<<numOfBlocks, THREADS_PER_BLOCK, 0, stream>>>(inputVec_d, outputVec_d, inputSize, numOfBlocks);
      reduceFinal<<<1, THREADS_PER_BLOCK, 0, stream>>>(outputVec_d, result_d, numOfBlocks);
    }

    double result_h = 0.0;
    cudaMemcpyAsync(&result_h, result_d, sizeof(double), cudaMemcpyDefault, stream);

    cudaStreamSynchronize(stream);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("[UsingStream] final reduced sum = %lf\n", result_h);
    printf("Execution time: %f (us)\n\n", (time * 1e-3f));
  }

  cudaStreamDestroy(stream);
}

int main(int argc, char **argv)
{
  size_t maxBlocks = 512;

  for (size_t size = 512; size <= 1<<27; size = size * 512) {

    printf("\n-----------------------------\n");
    printf("%zu elements\n", size);
    printf("Threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Launch iterations = %d\n", LAUNCH_ITERATIONS);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    inputVec_h = (float*) malloc(sizeof(float)*size);
    cudaMalloc(&inputVec_d, sizeof(float)*size);
    cudaMalloc(&outputVec_d, sizeof(double)*maxBlocks);
    cudaMalloc(&result_d, sizeof(double));

    init_input(inputVec_h, size);

    usingGraph(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
    usingStream(inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    cudaFree(inputVec_d);
    cudaFree(outputVec_d);
    cudaFree(result_d);
  }
  return EXIT_SUCCESS;
}
