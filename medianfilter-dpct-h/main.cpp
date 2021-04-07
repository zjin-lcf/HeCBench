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

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shrUtils.h"
#include "MedianFilter.cpp"

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

// Import host computation function 
extern "C" void MedianFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                 unsigned int uiWidth, unsigned int uiHeight);

void MedianFilterGPU(unsigned int *uiInputImage, unsigned int *uiOutputImage,
                     sycl::uchar4 *cmDevBufIn, unsigned int *cmDevBufOut,
                     const int uiImageWidth, const int uiImageHeight);

int main(int argc, char** argv)
{
  // Image data file
  const char* cPathAndName = argv[1]; 
  unsigned int uiImageWidth = 1920;   // Image width
  unsigned int uiImageHeight = 1080;  // Image height

  size_t szBuffBytes;                 // Size of main image buffers
  size_t szBuffWords;                 

  //char* cPathAndName = NULL;          // var for full paths to data, src, etc.
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

  sycl::uchar4 *cmDevBufIn;
  cmDevBufIn = (sycl::uchar4 *)dpct::dpct_malloc(szBuffBytes);

  unsigned int* cmDevBufOut;
  cmDevBufOut = (unsigned int *)dpct::dpct_malloc(szBuffBytes);

  // Warmup call 
  MedianFilterGPU (uiInput, uiOutput, cmDevBufIn, 
                   cmDevBufOut, uiImageWidth, uiImageHeight);

  // Process n loops on the GPU
  const int iCycles = 150;
  printf("\nRunning MedianFilterGPU for %d cycles...\n\n", iCycles);
  for (int i = 0; i < iCycles; i++)
  {
    MedianFilterGPU (uiInput, uiOutput, cmDevBufIn, 
                     cmDevBufOut, uiImageWidth, uiImageHeight);
  }

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
  dpct::dpct_free(cmDevBufIn);
  dpct::dpct_free(cmDevBufOut);

  if(bMatch == shrTRUE) 
    printf("PASS\n");
  else
    printf("FAIL\n");

  return EXIT_SUCCESS;
}

// Copies input data from host buf to the device, runs kernel, 
// copies output data back to output host buf
void MedianFilterGPU(unsigned int *uiInputImage, unsigned int *uiOutputImage,
                     sycl::uchar4 *cmDevBufIn, unsigned int *cmDevBufOut,
                     const int uiImageWidth, const int uiImageHeight)
{
  size_t szGlobalWorkSize[2];         // 2D global work items (ND range) for Median kernel
  size_t szLocalWorkSize[2];          // 2D local work items (work group) for Median kernel
  const int iBlockDimX = 16;
  const int iBlockDimY = 4;
  const int iLocalPixPitch = iBlockDimX + 2;

  dpct::dpct_memcpy(cmDevBufIn, (sycl::uchar4 *)uiInputImage,
                    uiImageWidth * uiImageHeight * sizeof(sycl::uchar4),
                    dpct::host_to_device);

  szLocalWorkSize[0] = iBlockDimX;
  szLocalWorkSize[1] = iBlockDimY;
  szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], uiImageWidth); 
  szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], uiImageHeight);

  sycl::range<3> lws(1, szLocalWorkSize[1], szLocalWorkSize[0]);
  sycl::range<3> gws(1, szGlobalWorkSize[1] / szLocalWorkSize[1],
                     szGlobalWorkSize[0] / szLocalWorkSize[0]);

  /*
  DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  {
    std::pair<dpct::buffer_t, size_t> cmDevBufIn_buf_ct0 =
        dpct::get_buffer_and_offset(cmDevBufIn);
    size_t cmDevBufIn_offset_ct0 = cmDevBufIn_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> cmDevBufOut_buf_ct1 =
        dpct::get_buffer_and_offset(cmDevBufOut);
    size_t cmDevBufOut_offset_ct1 = cmDevBufOut_buf_ct1.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint8_t, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          dpct_local_acc_ct1(sycl::range<1>(sizeof(sycl::uchar4) *
                                            iLocalPixPitch * (iBlockDimY + 2)),
                             cgh);
      auto cmDevBufIn_acc_ct0 =
          cmDevBufIn_buf_ct0.first.get_access<sycl::access::mode::read_write>(
              cgh);
      auto cmDevBufOut_acc_ct1 =
          cmDevBufOut_buf_ct1.first.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(sycl::nd_range<3>(gws * lws, lws),
                       [=](sycl::nd_item<3> item_ct1) {
                         const sycl::uchar4 *cmDevBufIn_ct0 =
                             (const sycl::uchar4 *)(&cmDevBufIn_acc_ct0[0] +
                                                    cmDevBufIn_offset_ct0);
                         unsigned int *cmDevBufOut_ct1 =
                             (unsigned int *)(&cmDevBufOut_acc_ct1[0] +
                                              cmDevBufOut_offset_ct1);
                         ckMedian(cmDevBufIn_ct0, cmDevBufOut_ct1,
                                  iLocalPixPitch, uiImageWidth, uiImageHeight,
                                  item_ct1, dpct_local_acc_ct1.get_pointer());
                       });
    });
  }

  dpct::dpct_memcpy((sycl::uchar4 *)uiOutputImage, cmDevBufOut,
                    uiImageWidth * uiImageHeight * sizeof(sycl::uchar4),
                    dpct::device_to_host);
}
