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

#include <chrono>
#include <memory>
#include <iostream>

typedef struct __attribute__((__aligned__(4)))
{
  unsigned char x;
  unsigned char y;
  unsigned char z;
  unsigned char w;
} uchar4;

typedef struct __attribute__((__aligned__(16)))
{
  float x;
  float y;
  float z;
  float w;
} float4;

inline uint DivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
//*****************************************************************
static float4 rgbaUintToFloat4(const unsigned int c)
{
  float4 rgba;
  rgba.x = c & 0xff;
  rgba.y = (c >> 8) & 0xff;
  rgba.z = (c >> 16) & 0xff;
  rgba.w = (c >> 24) & 0xff;
  return rgba;
}

static uchar4 rgbaUintToUchar4(unsigned int c)
{
    uchar4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

static unsigned int rgbaFloat4ToUint(const float4 rgba, const float fScale)
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

void BoxFilterHost( unsigned int *uiInput,
                    unsigned int *uiTmp,
                    unsigned int *uiDevOutput,
                    const int uiWidth,
                    const int uiHeight,
                    const int iRadius,
                    const float fScale)
{
  const int szMaxWorkgroupSize = 256;
  const int iRadiusAligned = ((iRadius + 15)/16) * 16;  // 16
  unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup

  if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
    uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

  const int uiBlockWidth = DivUp((size_t)uiWidth, (size_t)uiNumOutputPix);
  const int numTeams = uiHeight * uiBlockWidth;
  const int blockSize = iRadiusAligned + uiNumOutputPix + iRadius;

  for (int team = 0; team < numTeams; team++)
  {
    uchar4 uc4LocalData[iRadiusAligned+uiNumOutputPix+iRadius];
    for (int lid = 0; lid < blockSize; lid++)
    {
      int gidx = team % uiBlockWidth;
      int gidy = team / uiBlockWidth;

      int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
      int globalPosY = gidy;
      int iGlobalOffset = globalPosY * uiWidth + globalPosX;

      if (globalPosX >= 0 && globalPosX < uiWidth)
          uc4LocalData[lid] = rgbaUintToUchar4(uiInput[iGlobalOffset]);
      else
          uc4LocalData[lid] = {0, 0, 0, 0};
    }

    for (int lid = 0; lid < blockSize; lid++)
    {
      int gidx = team % uiBlockWidth;
      int gidy = team / uiBlockWidth;
      int globalPosX = gidx * uiNumOutputPix + lid - iRadiusAligned;
      int globalPosY = gidy;
      int iGlobalOffset = globalPosY * uiWidth + globalPosX;

      if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) &&
         (lid < (iRadiusAligned + (int)uiNumOutputPix)))
      {
          float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

          int iOffsetX = lid - iRadius;
          int iLimit = iOffsetX + (2 * iRadius) + 1;
          for(; iOffsetX < iLimit; iOffsetX++)
          {
              f4Sum.x += uc4LocalData[iOffsetX].x;
              f4Sum.y += uc4LocalData[iOffsetX].y;
              f4Sum.z += uc4LocalData[iOffsetX].z;
              f4Sum.w += uc4LocalData[iOffsetX].w;
          }

          uiTmp[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    }
  }

  for (int globalPosX = 0; globalPosX < uiWidth; globalPosX++) {
    unsigned int* uiInputImage = &uiTmp[globalPosX];
    unsigned int* uiOutputImage = &uiDevOutput[globalPosX];

    float4 f4Sum;
    float4 f4iRadius = {(float)iRadius, (float)iRadius, (float)iRadius, (float)iRadius};
    float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
    float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

    f4Sum = top_color * f4iRadius;
    for (int y = 0; y < iRadius + 1; y++)
    {
      if (y < uiHeight)
        f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
    }
    uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
    for(int y = 1; y < iRadius + 1; y++)
    {
      if (y + iRadius < uiHeight) {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= top_color;
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    }

    for(int y = iRadius + 1; y < uiHeight - iRadius; y++)
    {
      if (y + iRadius < uiHeight && y - iRadius >= 0) {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    }

    for (int y = uiHeight - iRadius; y < uiHeight; y++)
    {
      if (y < uiHeight && y - iRadius >= 0) {
        f4Sum += bot_color;
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    }
  }
}

