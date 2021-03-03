/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <memory>
#include <iostream>
#include <hip/hip_runtime.h>
#include "shrUtils.h"

extern void BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                          unsigned int uiWidth, unsigned int uiHeight, int r, float fScale );


const unsigned int iRadius = 10;                    // initial radius of 2D box filter mask
const float fScale = 1.0f/(2.0f * iRadius + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(const uint a, const uint b){
  return (a % b != 0) ? (a / b + 1) : (a / b);
}


// Helper function to convert float[4] rgba color to 32-bit unsigned integer
//*****************************************************************
__device__
float4 rgbaUintToFloat4(const unsigned int c)
{
  float4 rgba;
  rgba.x = c & 0xff;
  rgba.y = (c >> 8) & 0xff;
  rgba.z = (c >> 16) & 0xff;
  rgba.w = (c >> 24) & 0xff;
  return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
__device__
unsigned int rgbaFloat4ToUint(const float4 rgba, const float fScale)
{
  unsigned int uiPackedPix = 0U;
  uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
  uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
  uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
  uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
  return uiPackedPix;
}

__global__ void row_kernel (
    const uchar4* ucSource, 
    uint* uiDest,
    const unsigned int uiWidth, 
    const unsigned int uiHeight,
    const int iRadius,
    const int iRadiusAligned, 
    const float fScale, 
    const unsigned int uiNumOutputPix)
{

  HIP_DYNAMIC_SHARED( uchar4, uc4LocalData)

  int lid = threadIdx.x;
  int gidx = blockIdx.x;
  int gidy = blockIdx.y; 

  int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
  int globalPosY = gidy;
  int iGlobalOffset = globalPosY * uiWidth + globalPosX;

  // Read global data into LMEM
  if (globalPosX >= 0 && globalPosX < uiWidth)
  {
    uc4LocalData[lid] = ucSource[iGlobalOffset];
  }
  else
    uc4LocalData[lid] = {0, 0, 0, 0}; 

  __syncthreads();

  if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) && 
      (lid < (iRadiusAligned + (int)uiNumOutputPix)))
  {
    // Init summation registers to zero
    float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

    // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
    int iOffsetX = lid - iRadius;
    int iLimit = iOffsetX + (2 * iRadius) + 1;
    for(; iOffsetX < iLimit; iOffsetX++)
    {
      f4Sum.x += uc4LocalData[iOffsetX].x;
      f4Sum.y += uc4LocalData[iOffsetX].y;
      f4Sum.z += uc4LocalData[iOffsetX].z;
      f4Sum.w += uc4LocalData[iOffsetX].w; 
    }

    // Use inline function to scale and convert registers to packed RGBA values in a uchar4, 
    // and write back out to GMEM
    uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
  }
}

__global__ void col_kernel (
    const uint* uiSource, 
    uint* uiDest, 
    const unsigned int uiWidth, 
    const unsigned int uiHeight, 
    const int iRadius, 
    const float fScale)
{
  size_t globalPosX = blockIdx.x * blockDim.x + threadIdx.x;
  const uint* uiInputImage = &uiSource[globalPosX];
  uint* uiOutputImage = &uiDest[globalPosX];
  // do left edge
  float4 f4Sum;
  f4Sum = rgbaUintToFloat4(uiInputImage[0]) * float4(iRadius); 
  for (int y = 0; y < iRadius + 1; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
  }
  uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
  for(int y = 1; y < iRadius + 1; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
    f4Sum -= rgbaUintToFloat4(uiInputImage[0]);
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }

  // main loop
  for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
    f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }

  // do right edge
  for (int y = uiHeight - iRadius; y < uiHeight; y++) 
  {
    f4Sum += rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);
    f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
    uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
  }
}

void BoxFilterGPU (uchar4* cmBufIn,
    unsigned int* cmBufTmp,
    unsigned int* cmBufOut,
    const unsigned int uiWidth, 
    const unsigned int uiHeight, 
    const int r, 
    const float fScale )
{
  const int szMaxWorkgroupSize = 256;
  const int iRadiusAligned = ((r + 15)/16) * 16;  // 16
  unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup
  if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + r))
    uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - r;

  // Set global and local work sizes for row kernel // Workgroup padded left and right
  dim3 row_grid(DivUp((size_t)uiWidth, (size_t)uiNumOutputPix), uiHeight); 
  dim3 row_block((size_t)(iRadiusAligned + uiNumOutputPix + r), 1);

  // Launch row kernel
  hipLaunchKernelGGL(row_kernel, dim3(row_grid), dim3(row_block), sizeof(uchar4)*(iRadiusAligned+uiNumOutputPix+r), 0, 
      cmBufIn, cmBufTmp, uiWidth, uiHeight, iRadius, iRadiusAligned, fScale, uiNumOutputPix);

  // Set global and local work sizes for column kernel
  dim3 col_grid(DivUp((size_t)uiWidth, 64));
  dim3 col_block(64);

  // Launch column kernel
  hipLaunchKernelGGL(col_kernel, dim3(col_grid), dim3(col_block), 0, 0, cmBufTmp, cmBufOut, uiWidth, uiHeight, iRadius, fScale);
}

int main(int argc, char** argv)
{
  unsigned int uiImageWidth = 0;      // Image width
  unsigned int uiImageHeight = 0;     // Image height
  unsigned int* uiInput = NULL;       // Host buffer to hold input image data
  unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiDevOutput = NULL;      
  unsigned int* uiHostOutput = NULL;      

  shrLoadPPM4ub(argv[1], (unsigned char**)&uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %i, Height = %i, bpp = %i, Mask Radius = %i\n", 
      uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3, iRadius);
  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff= uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)malloc(szBuffBytes);
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);
  uiHostOutput = (unsigned int*)malloc(szBuffBytes);

  uchar4* cmDevBufIn;
  unsigned int* cmDevBufTmp;
  unsigned int* cmDevBufOut;

  hipMalloc((void**)&cmDevBufIn, szBuffBytes);
  hipMalloc((void**)&cmDevBufTmp, szBuffBytes);
  hipMalloc((void**)&cmDevBufOut, szBuffBytes);

  // Copy input data from host to device 
  hipMemcpy(cmDevBufIn, uiInput, szBuffBytes, hipMemcpyHostToDevice);

  // Warmup
  BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut, 
      uiImageWidth, uiImageHeight, iRadius, fScale);

  hipDeviceSynchronize();

  const int iCycles = 1000;
  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
  for (int i = 0; i < iCycles; i++)
  {
    BoxFilterGPU (cmDevBufIn, cmDevBufTmp, cmDevBufOut, 
        uiImageWidth, uiImageHeight, iRadius, fScale);
  }

  // Copy output from device to host
  hipMemcpy(uiDevOutput, cmDevBufOut, szBuffBytes, hipMemcpyDeviceToHost);

  hipFree(cmDevBufIn);
  hipFree(cmDevBufTmp);
  hipFree(cmDevBufOut);

  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, iRadius, fScale);

  // Verification 
  // The entire images do not match due to the difference between BoxFilterHostY and the column kernel )
  int error = 0;
  for (int i = iRadius * uiImageWidth; i < (uiImageHeight-iRadius)*uiImageWidth; i++)
  {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%d %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  if (error) 
    printf("FAIL\n");
  else
    printf("PASS\n");

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  return 0;
}

