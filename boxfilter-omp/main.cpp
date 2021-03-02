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
#include <omp.h>
#include "shrUtils.h"

typedef struct { float x; float y; float z; float w; } float4 ;
typedef struct { unsigned char x; unsigned char y; unsigned char z; unsigned char w; } uchar4 ;

extern
void BoxFilterHost( unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale );


const unsigned int iRadius = 10;                    // initial radius of 2D box filter mask
const float fScale = 1.0f/(2.0f * iRadius + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


#pragma omp declare target
// Helper function to convert float[4] rgba color to 32-bit unsigned integer
//*****************************************************************
float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

uchar4 rgbaUintToUchar4(unsigned int c)
{
    uchar4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

inline float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w};
}

inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline void operator-=(float4 &a, float4 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
#pragma omp end declare target

void BoxFilterGPU ( unsigned int *uiInput,
                    unsigned int *uiTmp,
                    unsigned int *uiDevOutput,
                    const unsigned int uiWidth, 
                    const unsigned int uiHeight, 
                    const int r, const float fScale )
{
    const int szMaxWorkgroupSize = 256;
    const int iRadiusAligned = ((r + 15)/16) * 16;  // 16
    unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup

    if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + r))
      uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - r;

    // Set team and thread sizes for row kernel // Workgroup padded left and right
    const int uiBlockWidth = DivUp((size_t)uiWidth, (size_t)uiNumOutputPix);
    const int numTeams = uiHeight * uiBlockWidth;
    const int blockSize = iRadiusAligned + uiNumOutputPix + r;

    // Launch row kernel
    #pragma omp target teams num_teams(numTeams) thread_limit(blockSize)
    {
      uchar4 uc4LocalData[90]; //16+64+10;
      #pragma omp parallel 
      {
        int lid = omp_get_thread_num(); 
        int gidx = omp_get_team_num() % uiBlockWidth;
        int gidy = omp_get_team_num() / uiBlockWidth;

        int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
        int globalPosY = gidy;
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        // Read global data into LMEM
        if (globalPosX >= 0 && globalPosX < uiWidth)
            // IBM xlc: no known conversion from 'unsigned int *' to 'uchar4 *' 
            uc4LocalData[lid] = rgbaUintToUchar4(uiInput[iGlobalOffset]);
        else
            uc4LocalData[lid] = {0, 0, 0, 0}; 

        #pragma omp barrier

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
            uiTmp[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
      }
    }

    // Launch column kernel
    #pragma omp target teams distribute parallel for thread_limit(64)
    for (size_t globalPosX = 0; globalPosX < uiWidth; globalPosX++) {
      unsigned int* uiInputImage = &uiTmp[globalPosX];
      unsigned int* uiOutputImage = &uiDevOutput[globalPosX];

      // do left edge
      float4 f4Sum;
      float4 f4iRadius = {iRadius, iRadius, iRadius, iRadius};
      f4Sum = rgbaUintToFloat4(uiInputImage[0]) * f4iRadius;
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
}

int main(int argc, char** argv)
{
    unsigned int uiImageWidth = 0;      // Image width
    unsigned int uiImageHeight = 0;     // Image height
    unsigned int* uiInput = NULL;       // Host buffer to hold input image data
    unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
    unsigned int* uiDevOutput = NULL;      
    unsigned int* uiHostOutput = NULL;      

    shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    printf("Image Width = %i, Height = %i, bpp = %i, Mask Radius = %i\n", 
           uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3, iRadius);
    printf("Using Local Memory for Row Processing\n\n");

    size_t szBuff= uiImageWidth * uiImageHeight;
    size_t szBuffBytes = szBuff * sizeof (unsigned int);

    // Allocate intermediate and output host image buffers
    uiTmp = (unsigned int*)malloc(szBuffBytes);
    uiDevOutput = (unsigned int*)malloc(szBuffBytes);
    uiHostOutput = (unsigned int*)malloc(szBuffBytes);

#pragma omp target data map(to: uiInput[0:szBuff]) \
                        map(alloc: uiTmp[0:szBuff]) \
                        map(from: uiDevOutput[0:szBuff])
{

    // Warmup
    BoxFilterGPU (uiInput, uiTmp, uiDevOutput, 
                  uiImageWidth, uiImageHeight, iRadius, fScale);

    const int iCycles = 1000;
    printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
    for (int i = 0; i < iCycles; i++)
    {
        BoxFilterGPU (uiInput, uiTmp, uiDevOutput,
                      uiImageWidth, uiImageHeight, iRadius, fScale);
    }
}

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

