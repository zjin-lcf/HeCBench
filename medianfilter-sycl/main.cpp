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
#include "common.h"
#include "shrUtils.h"
#include "MedianFilter.cpp"

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

// Import host computation function 
extern "C" void MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                 unsigned int uiWidth, unsigned int uiHeight);

double MedianFilterGPU(
    queue &q,
    unsigned int* uiInputImage, 
    unsigned int* uiOutputImage, 
    buffer<sycl::uchar4> &cmDevBufIn,
    buffer<unsigned int> &cmDevBufOut,
    const int uiImageWidth,
    const int uiImageHeight);

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <image file> <repeat>\n", argv[0]);
    return 1;
  }
  // Image data file
  const char* cPathAndName = argv[1]; 

  const int iCycles = atoi(argv[2]);

  unsigned int uiImageWidth = 1920;   // Image width
  unsigned int uiImageHeight = 1080;  // Image height

  size_t szBuffBytes;                 // Size of main image buffers
  size_t szBuffWords;                 

  unsigned int* uiInput;              // Host input buffer 
  unsigned int* uiOutput;             // Host output buffer

  // One device processes the whole image
  szBuffWords = uiImageHeight * uiImageWidth;
  szBuffBytes = szBuffWords * sizeof (unsigned int);

  uiInput = (unsigned int*) malloc (szBuffBytes);
  uiOutput = (unsigned int*) malloc (szBuffBytes);

  shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);

  printf("Image File\t = %s\nImage Dimensions = %u w x %u h x %lu bpp\n\n", 
         cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<sycl::uchar4, 1> cmDevBufIn(szBuffWords);
  buffer<unsigned int, 1> cmDevBufOut(szBuffWords);

  // Warmup call 
  MedianFilterGPU (q, uiInput, uiOutput, cmDevBufIn, 
                   cmDevBufOut, uiImageWidth, uiImageHeight);

  double time = 0.0;

  // Process n loops on the GPU
  printf("\nRunning MedianFilterGPU for %d cycles...\n\n", iCycles);
  for (int i = 0; i < iCycles; i++)
  {
    time += MedianFilterGPU (q, uiInput, uiOutput, cmDevBufIn, 
                             cmDevBufOut, uiImageWidth, uiImageHeight);
  }
  printf("Average kernel execution time: %f (s)\n\n", (time * 1e-9f) / iCycles);

  // Compute on host 
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  MedianFilterHost(uiInput, uiGolden, uiImageWidth, uiImageHeight);

  // Compare GPU and Host results:  Allow variance of 1 GV in up to 0.01% of pixels 
  printf("Comparing GPU Result to CPU Result...\n"); 
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.0001f);
  printf("\nGPU Result %s CPU Result within tolerance...\n", 
         (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

  // Cleanup and exit
  free(uiGolden);
  free(uiInput);
  free(uiOutput);

  if(bMatch == shrTRUE) 
    printf("PASS\n");
  else
    printf("FAIL\n");

  return EXIT_SUCCESS;
}

// Copies input data from host buf to the device, runs kernel, 
// copies output data back to output host buf
double MedianFilterGPU(
    queue &q,
    unsigned int* uiInputImage, 
    unsigned int* uiOutputImage, 
    buffer<sycl::uchar4> &cmDevBufIn,
    buffer<unsigned int> &cmDevBufOut,
    const int uiImageWidth,
    const int uiImageHeight)
{
  size_t szGlobalWorkSize[2];         // 2D global work items (ND range) for Median kernel
  size_t szLocalWorkSize[2];          // 2D local work items (work group) for Median kernel
  const int iBlockDimX = 16;
  const int iBlockDimY = 4;
  const int iLocalPixPitch = iBlockDimX + 2;

  q.submit([&] (handler &cgh) {
    auto acc = cmDevBufIn.get_access<sycl_discard_write>(cgh);
    cgh.copy((sycl::uchar4*)uiInputImage, acc);
  });

  szLocalWorkSize[0] = iBlockDimX;
  szLocalWorkSize[1] = iBlockDimY;
  szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], uiImageWidth); 
  szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], uiImageHeight);

  range<2> lws(szLocalWorkSize[1], szLocalWorkSize[0]);
  range<2> gws(szGlobalWorkSize[1], szGlobalWorkSize[0]);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (handler &cgh) {
    auto uc4Source = cmDevBufIn.get_access<sycl_read>(cgh);
    auto uiDest = cmDevBufOut.get_access<sycl_discard_write>(cgh);
    accessor<sycl::uchar4, 1, sycl_read_write, access::target::local> 
      uc4LocalData(iLocalPixPitch * (iBlockDimY + 2), cgh);
    cgh.parallel_for<class media_filter>(nd_range<2>(gws, lws), [=] (nd_item<2> item) {
      ckMedian(item, uc4Source.get_pointer(), uiDest.get_pointer(),
               uc4LocalData.get_pointer(), iLocalPixPitch, uiImageWidth, uiImageHeight);
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.submit([&] (handler &cgh) {
    auto acc = cmDevBufOut.get_access<sycl_read>(cgh);
    cgh.copy(acc, uiOutputImage);
  }).wait();

  return time;
}
