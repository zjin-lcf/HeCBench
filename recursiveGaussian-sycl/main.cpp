
#define CLAMP_TO_EDGE
#define MAC

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <chrono>
#include <sycl/sycl.hpp>
#include "main.h"
#include "shrUtils.h"

using float4 = sycl::float4;

// Inline device function to convert 32-bit unsigned integer to floating point rgba color
//*****************************************************************
float4 rgbaUintToFloat4(const unsigned int uiPackedRGBA)
{
    float4 rgba;
    rgba.x() = uiPackedRGBA & 0xff;
    rgba.y() = (uiPackedRGBA >> 8) & 0xff;
    rgba.z() = (uiPackedRGBA >> 16) & 0xff;
    rgba.w() = (uiPackedRGBA >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(const float4 rgba)
{
    unsigned int uiPackedRGBA = 0U;
    uiPackedRGBA |= 0x000000FF & (unsigned int)rgba.x();
    uiPackedRGBA |= 0x0000FF00 & (((unsigned int)rgba.y()) << 8);
    uiPackedRGBA |= 0x00FF0000 & (((unsigned int)rgba.z()) << 16);
    uiPackedRGBA |= 0xFF000000 & (((unsigned int)rgba.w()) << 24);
    return uiPackedRGBA;
}

// Transpose kernel (see transpose SDK sample for details)
//*****************************************************************
void Transpose(sycl::nd_item<2> &item,
               const unsigned int *uiDataIn,
               unsigned int *uiDataOut,
               const int iWidth, const int iHeight,
               unsigned int *uiLocalBuff)
{
    // read the matrix tile into LMEM
    unsigned int xIndex = item.get_global_id(1);
    unsigned int yIndex = item.get_global_id(0);

    if((xIndex < iWidth) && (yIndex < iHeight))
    {
        //uiLocalBuff[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = uiDataIn[(yIndex * iWidth) + xIndex];
        uiLocalBuff[item.get_local_id(0) * (item.get_local_range(1) + 1) +
                    item.get_local_id(1)] = uiDataIn[(yIndex * iWidth) + xIndex];
    }

    // Synchronize the read into LMEM
    item.barrier(sycl::access::fence_space::local_space);

    // write the transposed matrix tile to global memory
    // mul24(get_group_id(1), get_local_size(1)) + get_local_id(0);
    xIndex = sycl::mul24((unsigned)item.get_group(0), (unsigned)item.get_local_range(0)) + item.get_local_id(1);
    //mul24(get_group_id(0), get_local_size(0)) + get_local_id(1);
    yIndex = sycl::mul24((unsigned)item.get_group(1), (unsigned)item.get_local_range(1)) + item.get_local_id(0);
    if((xIndex < iHeight) && (yIndex < iWidth))
    {
        uiDataOut[(yIndex * iHeight) + xIndex] =
         //uiLocalBuff[get_local_id(0) * (get_local_size(1) + 1) + get_local_id(1)];
         uiLocalBuff[item.get_local_id(1) * (item.get_local_range(0) + 1) + item.get_local_id(0)];
    }
}

// 	simple 1st order recursive filter kernel
//*****************************************************************
//    - processes one image column per thread
//      parameters:
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data
//      iWidth  - image width
//      iHeight  - image height
//      a  - blur parameter
//*****************************************************************
void SimpleRecursiveRGBA(
  sycl::nd_item<1> &item,
  const unsigned int *uiDataIn,
  unsigned int *uiDataOut,
  const int iWidth, const int iHeight, const float a)
{
    // compute X pixel location and check in-bounds
  unsigned int X = item.get_global_id(0);
  if (X >= iWidth) return;

  // advance global pointers to correct column for this work item and x position
  uiDataIn += X;
  uiDataOut += X;

  // start forward filter pass
  float4 yp = rgbaUintToFloat4(*uiDataIn);  // previous output
  for (int Y = 0; Y < iHeight; Y++)
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * (float4)a;
    *uiDataOut = rgbaFloat4ToUint(yc);
    yp = yc;
    uiDataIn += iWidth;     // move to next row
    uiDataOut += iWidth;    // move to next row
  }

  // reset global pointers to point to last element in column for this work item and x position
  uiDataIn -= iWidth;
  uiDataOut -= iWidth;

  // start reverse filter pass: ensures response is symmetrical
  yp = rgbaUintToFloat4(*uiDataIn);
  for (int Y = iHeight - 1; Y > -1; Y--)
  {
    float4 xc = rgbaUintToFloat4(*uiDataIn);
    float4 yc = xc + (yp - xc) * (float4)a;
    *uiDataOut = rgbaFloat4ToUint((rgbaUintToFloat4(*uiDataOut) + yc) * 0.5f);
    yp = yc;
    uiDataIn -= iWidth;   // move to previous row
    uiDataOut -= iWidth;  // move to previous row
  }
}

// Recursive Gaussian filter
//*****************************************************************
//  parameters:
//      uiDataIn - pointer to input data (RGBA image packed into 32-bit integers)
//      uiDataOut - pointer to output data
//      iWidth  - image width
//      iHeight  - image height
//      a0-a3, b1, b2, coefp, coefn - filter parameters
//
//      If used, CLAMP_TO_EDGE is passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
void RecursiveRGBA(sycl::nd_item<1> &item,
                   const unsigned int *uiDataIn,
                   unsigned int *uiDataOut,
                   const int iWidth, const int iHeight,
                   const float a0, const float a1,
                   const float a2, const float a3,
                   const float b1, const float b2,
                   const float coefp, const float coefn)
{
    // compute X pixel location and check in-bounds
    //unsigned int X = mul24(get_group_id(0), get_local_size(0)) + get_local_id(0);
    unsigned int X = item.get_global_id(0);
	if (X >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    uiDataIn += X;
    uiDataOut += X;

    // start forward filter pass
    float4 xp = (float4)0.0f;  // previous input
    float4 yp = (float4)0.0f;  // previous output
    float4 yb = (float4)0.0f;  // previous output by 2

#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*uiDataIn);
    yb = xp * (float4)coefp;
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++)
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
        *uiDataOut = rgbaFloat4ToUint(yc);
        xp = xc;
        yb = yp;
        yp = yc;
        uiDataIn += iWidth;     // move to next row
        uiDataOut += iWidth;    // move to next row
    }

    // reset global pointers to point to last element in column for this work item and x position
    uiDataIn -= iWidth;
    uiDataOut -= iWidth;

    // start reverse filter pass: ensures response is symmetrical
    float4 xn = (float4)0.0f;
    float4 xa = (float4)0.0f;
    float4 yn = (float4)0.0f;
    float4 ya = (float4)0.0f;

#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*uiDataIn);
    xa = xn;
    yn = xn * (float4)coefn;
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y > -1; Y--)
    {
        float4 xc = rgbaUintToFloat4(*uiDataIn);
        float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn;
        xn = xc;
        ya = yn;
        yn = yc;
        *uiDataOut = rgbaFloat4ToUint(rgbaUintToFloat4(*uiDataOut) + yc);
        uiDataIn -= iWidth;   // move to previous row
        uiDataOut -= iWidth;  // move to previous row
    }
}

double GPUGaussianFilterRGBA(sycl::queue &q,
                             const unsigned int* uiInput,
                             unsigned int* uiOutput,
                             unsigned int *d_BufIn,
                             unsigned int *d_BufTmp,
                             unsigned int *d_BufOut,
                             const unsigned int uiImageWidth,
                             const unsigned int uiImageHeight,
                             const GaussParms* pGP)
{
#if USE_SIMPLE_FILTER
  float ema = pGP->ema;
#else
  float a0 = pGP->a0;
  float a1 = pGP->a1;
  float a2 = pGP->a2;
  float a3 = pGP->a3;
  float b1 = pGP->b1;
  float b2 = pGP->b2;
  float coefp = pGP->coefp;
  float coefn = pGP->coefn;
#endif

  unsigned int szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
  q.memcpy(d_BufIn, uiInput, szBuffBytes).wait();

  auto start = std::chrono::steady_clock::now();

  const int iTransposeBlockDim = 16;        // initial height and width dimension of 2D transpose workgroup
  size_t szGaussLocalWork = 256;
  size_t szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth);

  sycl::range<1> gauss_gws (szGaussGlobalWork);
  sycl::range<1> gauss_lws (szGaussLocalWork);

  q.submit([&] (sycl::handler &cgh) {
#if USE_SIMPLE_FILTER
    cgh.parallel_for<class simpleRecursiveRGBA>(
      sycl::nd_range<1>(gauss_gws, gauss_lws), [=] (sycl::nd_item<1> item) {
      SimpleRecursiveRGBA(item, d_BufIn, d_BufTmp, uiImageWidth, uiImageHeight, ema);
#else
    cgh.parallel_for<class recursiveRGBA>(
      sycl::nd_range<1>(gauss_gws, gauss_lws), [=] (sycl::nd_item<1> item) {
      RecursiveRGBA(item, d_BufIn, d_BufTmp, uiImageWidth, uiImageHeight,
                    a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
    });
  });

  size_t szTransposeGlobalWork[2];
  size_t szTransposeLocalWork[2] = {16, 16};
  // Launch transpose kernel in 1st direction
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth);
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight);
  sycl::range<2> t1_gws (szTransposeGlobalWork[1], szTransposeGlobalWork[0]);
  sycl::range<2> t1_lws (szTransposeLocalWork[1], szTransposeLocalWork[0]);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<unsigned int, 1>
      uiLocalBuff (sycl::range<1>(iTransposeBlockDim * (iTransposeBlockDim + 1)), cgh);
    cgh.parallel_for<class transpose1>(sycl::nd_range<2>(t1_gws, t1_lws), [=] (sycl::nd_item<2> item) {
      Transpose(item, d_BufTmp, d_BufOut, uiImageWidth, uiImageHeight, uiLocalBuff.get_pointer());
    });
  });

  // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
  // note width and height parameters flipped due to transpose
  szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight);

  sycl::range<1> gauss2_gws (szGaussGlobalWork);

  q.submit([&] (sycl::handler &cgh) {
#if USE_SIMPLE_FILTER
    cgh.parallel_for<class simpleRecursiveRGBA2>(
      sycl::nd_range<1>(gauss2_gws, gauss_lws), [=] (sycl::nd_item<1> item) {
      SimpleRecursiveRGBA(item, d_BufOut, d_BufTmp, uiImageHeight, uiImageWidth, ema);
#else
    cgh.parallel_for<class recursiveRGBA2>(
      sycl::nd_range<1>(gauss2_gws, gauss_lws), [=] (sycl::nd_item<1> item) {
      RecursiveRGBA(item, d_BufOut, d_BufTmp, uiImageHeight, uiImageWidth,
                    a0, a1, a2, a3, b1, b2, coefp, coefn);
#endif
    });
  });


  // Reset transpose global work dimensions and variable args
  // note width and height parameters flipped due to 1st transpose
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight);
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth);

  sycl::range<2> t2_gws (szTransposeGlobalWork[1], szTransposeGlobalWork[0]);
  //range<1> t2_lws (szTransposeLocalWork[1], szTransposeLobalWork[0]);
  // Launch transpose kernel in 2nd direction
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<unsigned int, 1>
      uiLocalBuff (sycl::range<1>(iTransposeBlockDim * (iTransposeBlockDim + 1)), cgh);
    cgh.parallel_for<class transpose2>(sycl::nd_range<2>(t2_gws, t1_lws), [=] (sycl::nd_item<2> item) {
      Transpose(item, d_BufTmp, d_BufOut, uiImageHeight, uiImageWidth, uiLocalBuff.get_pointer());
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  q.memcpy(uiOutput, d_BufOut, szBuffBytes).wait();
  return time;
}

int main(int argc, char** argv)
{
  if (argc != 3) {
    printf("Usage: %s <path to image> <repeat>\n", argv[0]);
    return 1;
  }

  const float fSigma = 10.0f;         // filter sigma (blur factor)
  const int iOrder = 0;               // filter order
  unsigned int uiImageWidth = 1920;   // Image width
  unsigned int uiImageHeight = 1080;  // Image height
  unsigned int* uiInput = NULL;       // Host buffer to hold input image data
  unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
  unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

  shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
  printf("Image Width = %i, Height = %i, bpp = %lu\n\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);
  const int iCycles = atoi(argv[2]);

  // Allocate intermediate and output host image buffers
  unsigned int szBuff = uiImageWidth * uiImageHeight;
  unsigned int szBuffBytes = szBuff * sizeof (unsigned int);
  uiTemp = (unsigned int*)malloc(szBuffBytes);
  uiOutput = (unsigned int*)malloc(szBuffBytes);
  printf("Allocate Host Image Buffers...\n");

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Allocate the source, intermediate and result buffer memory objects on the device GMEM
  unsigned int *d_BufIn= sycl::malloc_device<unsigned int>(szBuff, q);
  unsigned int *d_BufTmp= sycl::malloc_device<unsigned int>(szBuff, q);
  unsigned int *d_BufOut= sycl::malloc_device<unsigned int>(szBuff, q);

  // init filter coefficients
  PreProcessGaussParms (fSigma, iOrder, &GP);

  // Warmup call to assure OpenCL driver is awake
  GPUGaussianFilterRGBA(q, uiInput, uiOutput, d_BufIn, d_BufTmp,
                        d_BufOut, uiImageWidth, uiImageHeight, &GP);

  // Start round-trip timer and process iCycles loops on the GPU
  printf("\nRunning GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
  double time = 0.0;

  for (int i = 0; i < iCycles; i++)
  {
     time += GPUGaussianFilterRGBA(q, uiInput, uiOutput, d_BufIn, d_BufTmp,
                                   d_BufOut, uiImageWidth, uiImageHeight, &GP);
  }

  printf("Average execution time of kernels: %f (s)\n", (time * 1e-9f) / iCycles);

  // Compute on host
  unsigned int* uiGolden = (unsigned int*)malloc(szBuffBytes);
  HostRecursiveGaussianRGBA(uiInput, uiTemp, uiGolden, uiImageWidth, uiImageHeight, &GP);

  printf("Comparing GPU Result to CPU Result...\n");
  shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.01f);
  printf("\nGPU Result %s CPU Result within tolerance...\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match");

  free(uiGolden);
  free(uiInput);
  free(uiTemp);
  free(uiOutput);
  sycl::free(d_BufIn, q);
  sycl::free(d_BufTmp, q);
  sycl::free(d_BufOut, q);
  return 0;
}
