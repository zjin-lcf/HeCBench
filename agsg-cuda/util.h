#ifndef __UTILS__
#define __UTILS__

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "defines.h"

__host__ int iDivUp(int a, int b);
__host__ int iDivDown(int a, int b);
__host__ int iAlignUp(int a, int b);
__host__ int iAlignDown(int a, int b);

__host__ int iDivUpF(int a, float b);
__host__ int iClosestPowerOfTwo(int x);

__host__ void Float4ToUchar4(float4 *inputImage, uchar4 *outputImage, int width, int height);
__global__ void float4toUchar4(const float4 *__restrict__ inputImage, uchar4 *__restrict__ outputImage, int width, int height);

__host__ void Uchar4ToFloat4(uchar4 *inputImage, float4 *outputImage, int width, int height);
__global__ void uchar4tofloat4(const uchar4 *__restrict__ inputImage, float4 *__restrict__ outputImage, int width, int height);

#endif
