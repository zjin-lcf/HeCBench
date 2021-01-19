#include <stdio.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda.h>
#include "constants.h"

#define N_THREADS_PER_BLOCK 256

__global__ void find_min_max_u_kernel(
    const float *__restrict__ g_u, float *__restrict__ g_max, float *__restrict__ g_min
) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int tidFromBack = blockDim.x - 1 - tid;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_u[i];
    __syncthreads();

    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            if (sdata[tid + s] > sdata[tid])
            {
                sdata[tid] = sdata[tid + s];
            }
        }
        if (tidFromBack < s) {
            if (sdata[tid - s] < sdata[tid]) {
                sdata[tid] = sdata[tid - s];
            }
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        g_max[blockIdx.x] = sdata[0];
    }
    if (tidFromBack == 0)
    {
        g_min[blockIdx.x] = sdata[tid];
    }
}

void find_min_max_u_cuda(
    const float *__restrict__ u, llint u_size, float *__restrict__ min_u, float *__restrict__ max_u
) {
    llint u_block = u_size / N_THREADS_PER_BLOCK;
    llint u_remainder = u_size % N_THREADS_PER_BLOCK;

    llint d_block = u_block;
    if (u_remainder != 0) { d_block += 1; }
    llint d_size = d_block * N_THREADS_PER_BLOCK;

    llint reminder_size = N_THREADS_PER_BLOCK - u_remainder;
    float *reminder = (float *)malloc(reminder_size * sizeof(float));
    memcpy(reminder, u, reminder_size * sizeof(float));

    float* max = (float*)malloc(d_block * sizeof(float));
    float *min = (float*)malloc(d_block * sizeof(float));

    float* d_u, * d_max, * d_min;
    cudaMalloc(&d_u, d_size * sizeof(float));
    cudaMalloc(&d_max, d_block * sizeof(float));
    cudaMalloc(&d_min, d_block * sizeof(float));

    cudaMemcpy(d_u, u, u_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u+u_size, reminder, reminder_size * sizeof(float), cudaMemcpyHostToDevice);
    find_min_max_u_kernel<<<d_block, N_THREADS_PER_BLOCK, sizeof(float) * N_THREADS_PER_BLOCK>>>(d_u, d_max, d_min);
    cudaMemcpy(max, d_max, d_block * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(min, d_min, d_block * sizeof(float), cudaMemcpyDeviceToHost);

    *min_u = FLT_MAX, *max_u = FLT_MIN;
    for (size_t i = 0; i < d_block; i++) {
        *min_u = fminf(*min_u, min[i]);
        *max_u = fmaxf(*max_u, max[i]);
    }

    cudaFree(d_max);
    cudaFree(d_min);
    cudaFree(d_u);
    free(reminder);
    free(max);
    free(min);
}
