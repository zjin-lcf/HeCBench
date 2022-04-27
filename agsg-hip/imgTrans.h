#ifndef __IMG_TRANS__
#define __IMG_TRANS__

#include <hip/hip_runtime.h>
#include "defines.h"

__host__ void Rgb2CIELab(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2CIELabNative(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2CIELabImproved(const uchar4* __restrict__ inputImg,float4* outputImg, int width, int height);

__host__ void Rgb2XYZ(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2XYZNative(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2XYZImproved(const uchar4* __restrict__ inputImg, float4* outputImg, int width, int height);

__host__ void Rgb2HSV(uchar4* inputImg, float4* outputImg, int width, int height);
__global__ void kRgb2HSV(const uchar4* __restrict__ inputImg, float4* outputImg, int width, int height);

#endif
