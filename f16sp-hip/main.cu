/**
 * Copyright 2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <hip/hip_fp16.h>

#define NUM_OF_BLOCKS 1024
#define NUM_OF_THREADS 128

__forceinline__ __device__ 
void reduceInShared_intrinsics(half2 * const v)
{
    int lid = threadIdx.x;	
    if(lid<64) v[lid] = __hadd2( v[lid], v[lid+64]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+32]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+16]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+8]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+4]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+2]);
    __syncthreads();
    if(lid<32) v[lid] = __hadd2( v[lid], v[lid+1]);
    __syncthreads();
}

__forceinline__ __device__
void reduceInShared_native(half2 * const v)
{
    int lid = threadIdx.x;	
    if(lid<64) v[lid] = v[lid] + v[lid+64];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+32];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+16];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+8];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+4];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+2];
    __syncthreads();
    if(lid<32) v[lid] = v[lid] + v[lid+1];
    __syncthreads();
}

__global__
void scalarProductKernel_intrinsics(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value = __float2half2_rn(0.f);

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
    {
        value = __hfma2(a[i], b[i], value);
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_intrinsics(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
        float f_result = __low2float(result) + __high2float(result);
        results[blockIdx.x] = f_result;
    }
}

__global__
void scalarProductKernel_native(
        half2 const * const a,
        half2 const * const b,
        float * const results,
        size_t const size
        )
{
    const int stride = gridDim.x*blockDim.x;
    __shared__ half2 shArray[NUM_OF_THREADS];

    half2 value(0.f, 0.f);

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i+=stride)
    {
        value = a[i] * b[i] + value;
    }

    shArray[threadIdx.x] = value;
    __syncthreads();
    reduceInShared_native(shArray);

    if (threadIdx.x == 0)
    {
        half2 result = shArray[0];
	// x and y are protected members
        //float f_result = (float)result.y + (float)result.x;
        float f_result = (float)__half2_raw(result).data.x +
                         (float)__half2_raw(result).data.y;
        results[blockIdx.x] = f_result;
    }
}

// temp.x and temp.y are protected members
void generateInput(half2 * a, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        a[i] = half2( static_cast<float>(rand() % 4),
	              static_cast<float>(rand() % 2) );
    }
}

int main(int argc, char *argv[])
{
    size_t size = NUM_OF_BLOCKS*NUM_OF_THREADS*16;

    half2 * a, *b;
    half2 * d_a, *d_b;

    float * r;  // result
    float * d_r;

    a = (half2*) malloc (size*sizeof(half2));
    b = (half2*) malloc (size*sizeof(half2));
    hipMalloc((void**)&d_a, size*sizeof(half2));
    hipMalloc((void**)&d_b, size*sizeof(half2));

    r = (float*) malloc (NUM_OF_BLOCKS*sizeof(float));
    hipMalloc((void**)&d_r, NUM_OF_BLOCKS*sizeof(float));

    srand(123); 
    generateInput(a, size);
    hipMemcpy(d_a, a, size*sizeof(half2), hipMemcpyHostToDevice);

    generateInput(b, size);
    hipMemcpy(d_b, b, size*sizeof(half2), hipMemcpyHostToDevice);

    for (int i = 0; i < 10000; i++)
      hipLaunchKernelGGL(scalarProductKernel_intrinsics, dim3(NUM_OF_BLOCKS), dim3(NUM_OF_THREADS), 0, 0, d_a, d_b, d_r, size);

    hipMemcpy(r, d_r, NUM_OF_BLOCKS*sizeof(float), hipMemcpyDeviceToHost);


    float result_intrinsics = 0;
    for (int i = 0; i < NUM_OF_BLOCKS; ++i)
    {
        result_intrinsics += r[i];
    }
    printf("Result intrinsics\t: %f \n", result_intrinsics);

    for (int i = 0; i < 10000; i++)
      hipLaunchKernelGGL(scalarProductKernel_native, dim3(NUM_OF_BLOCKS), dim3(NUM_OF_THREADS), 0, 0, d_a, d_b, d_r, size);

    hipMemcpy(r, d_r, NUM_OF_BLOCKS*sizeof(float), hipMemcpyDeviceToHost);

    float result_native = 0;
    for (int i = 0; i < NUM_OF_BLOCKS; ++i)
    {
        result_native += r[i];
    }
    printf("Result native operators\t: %f \n", result_native);


    printf("fp16ScalarProduct %s\n", (fabs(result_intrinsics - result_native) < 0.00001) ? 
                                     "PASSED" : "FAILED");

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_r);
    free(a);
    free(b);
    free(r);

    return EXIT_SUCCESS;
}
