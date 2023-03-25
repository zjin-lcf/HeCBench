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
#include "common.h"
#include "shrUtils.h"

extern
void BoxFilterHost( unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale );


const unsigned int RADIUS = 10;                    // initial radius of 2D box filter mask
const float SCALE = 1.0f/(2.0f * RADIUS + 1.0f);  // precalculated GV rescaling value

inline uint DivUp(const uint a, const uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

// Helper function to convert float[4] rgba color to 32-bit unsigned integer
sycl::float4 rgbaUintToFloat4(const unsigned int c)
{
    sycl::float4 rgba;
    rgba.x() = c & 0xff;
    rgba.y() = (c >> 8) & 0xff;
    rgba.z() = (c >> 16) & 0xff;
    rgba.w() = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
unsigned int rgbaFloat4ToUint(const sycl::float4 rgba, const float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x() * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y() * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z() * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w() * fScale)) << 24);
    return uiPackedPix;
}

void BoxFilterGPU ( queue &q, 
                    buffer<sycl::uchar4, 1> &cmBufIn,
                    buffer<unsigned int, 1> &cmBufTmp,
                    buffer<unsigned int, 1> &cmBufOut,
                    const unsigned int uiWidth, 
                    const unsigned int uiHeight, 
                    const int iRadius, const float fScale )
{
    const int szMaxWorkgroupSize = 256;
    const int iRadiusAligned = ((iRadius + 15)/16) * 16;  // 16
    unsigned int uiNumOutputPix = 64;  // Default output pix per workgroup
    if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + iRadius))
      uiNumOutputPix = szMaxWorkgroupSize - iRadiusAligned - iRadius;

    // Set global and local work sizes for row kernel // Workgroup padded left and right
    range<2> row_gws(uiHeight, (size_t)(iRadiusAligned + uiNumOutputPix + iRadius) * 
                                      DivUp((size_t)uiWidth, (size_t)uiNumOutputPix));
    range<2> row_lws(1, (size_t)(iRadiusAligned + uiNumOutputPix + iRadius));

    // Launch row kernel
    q.submit([&] (handler &cgh) {
    auto ucSource = cmBufIn.get_access<sycl_read>(cgh);
    auto uiDest = cmBufTmp.get_access<sycl_discard_write>(cgh);
    accessor<sycl::uchar4, 1, sycl_read_write, access::target::local> 
      uc4LocalData(iRadiusAligned + uiNumOutputPix + iRadius, cgh);
    cgh.parallel_for<class row_kernel>(nd_range<2>(row_gws, row_lws), [=] (nd_item<2> item) {
        int lid = item.get_local_id(1);
        int gidx = item.get_group(1);
        int gidy = item.get_group(0);

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

        item.barrier(access::fence_space::local_space);

        if((globalPosX >= 0) && (globalPosX < uiWidth) && (lid >= iRadiusAligned) && 
           (lid < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            // Init summation registers to zero
            sycl::float4 f4Sum = {0.0f, 0.0f, 0.0f, 0.0f};

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            int iOffsetX = lid - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x() += uc4LocalData[iOffsetX].x();
                f4Sum.y() += uc4LocalData[iOffsetX].y();
                f4Sum.z() += uc4LocalData[iOffsetX].z();
                f4Sum.w() += uc4LocalData[iOffsetX].w(); 
            }

            // Use inline function to scale and convert registers to packed RGBA values in a sycl::uchar4, 
            // and write back out to GMEM
            uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
      });
    });

    // Set global and local work sizes for column kernel
    range<1> col_gws(64 * DivUp((size_t)uiWidth, 64));
    range<1> col_lws(64);

    // Launch column kernel
    q.submit([&] (handler &cgh) {
    auto uiSource = cmBufTmp.get_access<sycl_read>(cgh);
    auto uiDest = cmBufOut.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class col_kernel>(nd_range<1>(col_gws, col_lws), [=] (nd_item<1> item) {
      size_t globalPosX = item.get_global_id(0);
      auto uiInputImage = uiSource.get_pointer() + globalPosX;
      auto uiOutputImage = uiDest.get_pointer() + globalPosX;

      sycl::float4 top_color = rgbaUintToFloat4(uiInputImage[0]);
      sycl::float4 bot_color = rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);

      sycl::float4 radius = {iRadius, iRadius, iRadius, iRadius};
      sycl::float4 f4Sum = top_color * radius ;
      for (int y = 0; y < iRadius + 1; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
      }
      uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
      for(int y = 1; y < iRadius + 1; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
          f4Sum -= top_color;
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
      
      for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
      {
          f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
          f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }

      for (int y = uiHeight - iRadius; y < uiHeight; y++) 
      {
          f4Sum += bot_color;
          f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
          uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
      }
    });
   });
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage %s <PPM image> <repeat>\n", argv[0]);
    return 1;
  }
  unsigned int uiImageWidth = 0;     // Image width
  unsigned int uiImageHeight = 0;    // Image height
  unsigned int* uiInput = NULL;      // Host buffer to hold input image data
  unsigned int* uiTmp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiDevOutput = NULL;      
  unsigned int* uiHostOutput = NULL;      

  shrLoadPPM4ub(argv[1], (uchar **)&uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %u, Height = %u, bpp = %u, Mask Radius = %u\n", 
         uiImageWidth, uiImageHeight, unsigned(sizeof(unsigned int) * 8), RADIUS);
  printf("Using Local Memory for Row Processing\n\n");

  size_t szBuff= uiImageWidth * uiImageHeight;
  size_t szBuffBytes = szBuff * sizeof (unsigned int);

  // Allocate intermediate and output host image buffers
  uiTmp = (unsigned int*)malloc(szBuffBytes);
  uiDevOutput = (unsigned int*)malloc(szBuffBytes);
  uiHostOutput = (unsigned int*)malloc(szBuffBytes);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<sycl::uchar4, 1> cmDevBufIn(szBuff);
  buffer<unsigned int, 1> cmDevBufTmp(szBuff);
  buffer<unsigned int, 1> cmDevBufOut(szBuff);

  // Copy input data from host to device 
  q.submit([&] (handler &cgh) {
    auto input = cmDevBufIn.get_access<sycl_write>(cgh);
    cgh.copy((sycl::uchar4*)uiInput, input);
  });

  // Warmup
  BoxFilterGPU (q, cmDevBufIn, cmDevBufTmp, cmDevBufOut,
                uiImageWidth, uiImageHeight, RADIUS, SCALE);
  q.wait();

  const int iCycles = atoi(argv[2]);
  printf("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iCycles; i++)
  {
    BoxFilterGPU (q, cmDevBufIn, cmDevBufTmp, cmDevBufOut, 
        uiImageWidth, uiImageHeight, RADIUS, SCALE);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average device execution time %f (us)\n", (time * 1e-3f) / iCycles);

  // Copy output from device to host
  q.submit([&] (handler &cgh) {
    auto output = cmDevBufOut.get_access<sycl_read>(cgh);
    cgh.copy(output, uiDevOutput);
  }).wait();

  // Do filtering on the host
  BoxFilterHost(uiInput, uiTmp, uiHostOutput, uiImageWidth, uiImageHeight, RADIUS, SCALE);

  // Verification 
  // The entire images do not match due to the difference between BoxFilterHostY and the column kernel )
  int error = 0;
  for (unsigned i = RADIUS * uiImageWidth; i < (uiImageHeight-RADIUS)*uiImageWidth; i++)
  {
    if (uiDevOutput[i] != uiHostOutput[i]) {
      printf("%d %08x %08x\n", i, uiDevOutput[i], uiHostOutput[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(uiInput);
  free(uiTmp);
  free(uiDevOutput);
  free(uiHostOutput);
  return 0;
}
