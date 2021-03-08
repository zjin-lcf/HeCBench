
#define CLAMP_TO_EDGE 
#define MAC

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <iostream>
#include <cassert>
#include "main.h"
#include "shrUtils.h"

// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************

sycl::float4 rgbaUintToFloat4(const unsigned int uiPackedRGBA)
{
    sycl::float4 rgba;
    rgba.x() = uiPackedRGBA & 0xff;
    rgba.y() = (uiPackedRGBA >> 8) & 0xff;
    rgba.z() = (uiPackedRGBA >> 16) & 0xff;
    rgba.w() = (uiPackedRGBA >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************

unsigned int rgbaFloat4ToUint(const sycl::float4 rgba)
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

void Transpose(const unsigned int* uiDataIn, 
               unsigned int* uiDataOut, 
               const int iWidth, const int iHeight, sycl::nd_item<3> item_ct1,
               unsigned int *uiLocalBuff)
{
    // read the matrix tile into LMEM
    unsigned int xIndex =
        item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
        item_ct1.get_local_id(2);
    unsigned int yIndex =
        item_ct1.get_group(1) * item_ct1.get_local_range().get(1) +
        item_ct1.get_local_id(1);

    if((xIndex < iWidth) && (yIndex < iHeight))
    {
        //uiLocalBuff[get_local_id(1) * (get_local_size(0) + 1) + get_local_id(0)] = uiDataIn[(yIndex * iWidth) + xIndex];
        uiLocalBuff[item_ct1.get_local_id(1) *
                        (item_ct1.get_local_range().get(2) + 1) +
                    item_ct1.get_local_id(2)] =
            uiDataIn[(yIndex * iWidth) + xIndex];
    }

    // Synchronize the read into LMEM
    item_ct1.barrier();

    // write the transposed matrix tile to global memory
    xIndex = sycl::mul24((int)item_ct1.get_group(1),
                         (int)item_ct1.get_local_range(1)) +
             item_ct1.get_local_id(2);
    yIndex = sycl::mul24((int)item_ct1.get_group(2),
                         (int)item_ct1.get_local_range(2)) +
             item_ct1.get_local_id(1);
    if((xIndex < iHeight) && (yIndex < iWidth))
    {
        uiDataOut[(yIndex * iHeight) + xIndex] =
            // uiLocalBuff[get_local_id(0) * (get_local_size(1) + 1) +
            // get_local_id(1)];
            uiLocalBuff[item_ct1.get_local_id(2) *
                            (item_ct1.get_local_range().get(1) + 1) +
                        item_ct1.get_local_id(1)];
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
  const unsigned int* uiDataIn,
  unsigned int* uiDataOut,
  const int iWidth, const int iHeight, const float a, sycl::nd_item<3> item_ct1)
{
    // compute X pixel location and check in-bounds
  unsigned int X = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                   item_ct1.get_local_id(2);
  if (X >= iWidth) return;
    
  // advance global pointers to correct column for this work item and x position
  uiDataIn += X;    
  uiDataOut += X;

  // start forward filter pass
  sycl::float4 yp = rgbaUintToFloat4(*uiDataIn); // previous output
  for (int Y = 0; Y < iHeight; Y++) 
  {
    sycl::float4 xc = rgbaUintToFloat4(*uiDataIn);
    sycl::float4 yc = xc + (yp - xc) * sycl::float4(a);
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
    sycl::float4 xc = rgbaUintToFloat4(*uiDataIn);
    sycl::float4 yc = xc + (yp - xc) * sycl::float4(a);
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
void RecursiveRGBA(
                   const unsigned int* uiDataIn, 
                   unsigned int* uiDataOut, 
                   const int iWidth, const int iHeight, 
                   const float a0, const float a1, 
                   const float a2, const float a3, 
                   const float b1, const float b2, 
                   const float coefp, const float coefn,
                   sycl::nd_item<3> item_ct1)
{
    // compute X pixel location and check in-bounds
    //unsigned int X = mul24(get_group_id(0), get_local_size(0)) + get_local_id(0);
    unsigned int X = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
    if (X >= iWidth) return;

    // advance global pointers to correct column for this work item and x position
    uiDataIn += X;    
    uiDataOut += X;

    // start forward filter pass
    sycl::float4 xp = sycl::float4(0.0f); // previous input
    sycl::float4 yp = sycl::float4(0.0f); // previous output
    sycl::float4 yb = sycl::float4(0.0f); // previous output by 2

#ifdef CLAMP_TO_EDGE
    xp = rgbaUintToFloat4(*uiDataIn); 
    yb = xp * sycl::float4(coefp);
    yp = yb;
#endif

    for (int Y = 0; Y < iHeight; Y++) 
    {
        sycl::float4 xc = rgbaUintToFloat4(*uiDataIn);
        sycl::float4 yc = (xc * a0) + (xp * a1) - (yp * b1) - (yb * b2);
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
    sycl::float4 xn = sycl::float4(0.0f);
    sycl::float4 xa = sycl::float4(0.0f);
    sycl::float4 yn = sycl::float4(0.0f);
    sycl::float4 ya = sycl::float4(0.0f);

#ifdef CLAMP_TO_EDGE
    xn = rgbaUintToFloat4(*uiDataIn);
    xa = xn; 
    yn = xn * sycl::float4(coefn);
    ya = yn;
#endif

    for (int Y = iHeight - 1; Y > -1; Y--) 
    {
        sycl::float4 xc = rgbaUintToFloat4(*uiDataIn);
        sycl::float4 yc = (xn * a2) + (xa * a3) - (yn * b1) - (ya * b2);
        xa = xn; 
        xn = xc; 
        ya = yn; 
        yn = yc;
        *uiDataOut = rgbaFloat4ToUint(rgbaUintToFloat4(*uiDataOut) + yc);
        uiDataIn -= iWidth;   // move to previous row
        uiDataOut -= iWidth;  // move to previous row
    }
}

void GPUGaussianFilterRGBA(const unsigned int* uiInput,
                           unsigned int* uiOutput,
                           unsigned int* d_BufIn,
                           unsigned int* d_BufTmp,
                           unsigned int* d_BufOut,
                           const unsigned int uiImageWidth,
                           const unsigned int uiImageHeight, 
                           const GaussParms* pGP)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
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
  q_ct1.memcpy(d_BufIn, uiInput, szBuffBytes).wait();

  // const int iTransposeBlockDim = 16; // initial height and width dimension of 2D transpose workgroup 
  size_t szGaussLocalWork = 256;
  size_t szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageWidth);
  sycl::range<3> g_grid(1, 1, szGaussGlobalWork / szGaussLocalWork);
  sycl::range<3> g_block(1, 1, szGaussLocalWork);

#if USE_SIMPLE_FILTER
  SimpleRecursiveRGBA<<<g_grid, g_block>>>(d_BufIn, d_BufTmp, uiImageWidth, uiImageHeight, ema);
#else
        /*
        DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(g_grid * g_block, g_block),
                     [=](sycl::nd_item<3> item_ct1) {
                       RecursiveRGBA(d_BufIn, d_BufTmp, uiImageWidth,
                                     uiImageHeight, a0, a1, a2, a3, b1, b2,
                                     coefp, coefn, item_ct1);
                     });
  });
#endif

  size_t szTransposeGlobalWork[2];
  size_t szTransposeLocalWork[2] = {16, 16};
  // Launch transpose kernel in 1st direction
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageWidth); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageHeight);
  sycl::range<3> t1_grid(1, szTransposeGlobalWork[1] / szTransposeLocalWork[1],
                         szTransposeGlobalWork[0] / szTransposeLocalWork[0]);
  sycl::range<3> t1_block(1, szTransposeLocalWork[1], szTransposeLocalWork[0]);

  /*
  DPCT1049:1: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        uiLocalBuff_acc_ct1(sycl::range<1>(272 /*16*17*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(t1_grid * t1_block, t1_block),
                     [=](sycl::nd_item<3> item_ct1) {
                       Transpose(d_BufTmp, d_BufOut, uiImageWidth,
                                 uiImageHeight, item_ct1,
                                 uiLocalBuff_acc_ct1.get_pointer());
                     });
  });

  // Reset Gaussian global work dimensions and variable args, then process in 2nd dimension
  // note width and height parameters flipped due to transpose
  szGaussGlobalWork = shrRoundUp((int)szGaussLocalWork, uiImageHeight);

  sycl::range<3> g2_grid(1, 1, szGaussGlobalWork / szGaussLocalWork);

#if USE_SIMPLE_FILTER
  SimpleRecursiveRGBA<<<g2_grid, g_block>>>(d_BufOut, d_BufTmp, uiImageHeight, uiImageWidth, ema);
#else
        /*
        DPCT1049:2: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
  q_ct1.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<3>(g2_grid * g_block, g_block),
                     [=](sycl::nd_item<3> item_ct1) {
                       RecursiveRGBA(d_BufOut, d_BufTmp, uiImageHeight,
                                     uiImageWidth, a0, a1, a2, a3, b1, b2,
                                     coefp, coefn, item_ct1);
                     });
  });
#endif

  // Reset transpose global work dimensions and variable args 
  // note width and height parameters flipped due to 1st transpose
  szTransposeGlobalWork[0] = shrRoundUp((int)szTransposeLocalWork[0], uiImageHeight); 
  szTransposeGlobalWork[1] = shrRoundUp((int)szTransposeLocalWork[1], uiImageWidth);

  sycl::range<3> t2_grid(1, szTransposeGlobalWork[1] / szTransposeLocalWork[1],
                         szTransposeGlobalWork[0] / szTransposeLocalWork[0]);
  //range<1> t2_lws (szTransposeLocalWork[1], szTransposeLobalWork[0]);
  // Launch transpose kernel in 2nd direction
  /*
  DPCT1049:3: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                   sycl::access::target::local>
        uiLocalBuff_acc_ct1(sycl::range<1>(272 /*16*17*/), cgh);

    cgh.parallel_for(sycl::nd_range<3>(t2_grid * t1_block, t1_block),
                     [=](sycl::nd_item<3> item_ct1) {
                       Transpose(d_BufTmp, d_BufOut, uiImageHeight,
                                 uiImageWidth, item_ct1,
                                 uiLocalBuff_acc_ct1.get_pointer());
                     });
  });

  q_ct1.memcpy(uiOutput, d_BufOut, szBuffBytes).wait();
}

int main(int argc, char** argv)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
    const float fSigma = 10.0f;         // filter sigma (blur factor)
    const int iOrder = 0;               // filter order
    unsigned int uiImageWidth = 1920;   // Image width
    unsigned int uiImageHeight = 1080;  // Image height
    unsigned int* uiInput = NULL;       // Host buffer to hold input image data
    unsigned int* uiTemp = NULL;        // Host buffer to hold intermediate image data
    unsigned int* uiOutput = NULL;      // Host buffer to hold output image data

    shrLoadPPM4ub(argv[1], (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    printf("Image Width = %i, Height = %i, bpp = %lu\n\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

    // Allocate intermediate and output host image buffers
    unsigned int szBuff = uiImageWidth * uiImageHeight;
    unsigned int szBuffBytes = szBuff * sizeof (unsigned int);
    uiTemp = (unsigned int*)malloc(szBuffBytes);
    uiOutput = (unsigned int*)malloc(szBuffBytes);
    printf("Allocate Host Image Buffers...\n"); 

    // Allocate the source, intermediate and result buffer memory objects on the device GMEM
    unsigned int* d_BufIn;
    unsigned int* d_BufTmp;
    unsigned int* d_BufOut;
    d_BufIn = (unsigned int *)sycl::malloc_device(szBuffBytes, q_ct1);
    d_BufTmp = (unsigned int *)sycl::malloc_device(szBuffBytes, q_ct1);
    d_BufOut = (unsigned int *)sycl::malloc_device(szBuffBytes, q_ct1);

    // init filter coefficients
    PreProcessGaussParms (fSigma, iOrder, &GP);

    // Warmup call to assure OpenCL driver is awake
    GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp, d_BufOut, uiImageWidth, uiImageHeight, &GP);

    // Start round-trip timer and process iCycles loops on the GPU
    const int iCycles = 150;
    printf("\nRunning GPUGaussianFilterRGBA for %d cycles...\n\n", iCycles);
    for (int i = 0; i < iCycles; i++)
    {
       GPUGaussianFilterRGBA(uiInput, uiOutput, d_BufIn, d_BufTmp, d_BufOut, uiImageWidth, uiImageHeight, &GP);
    }

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
    sycl::free(d_BufIn, q_ct1);
    sycl::free(d_BufTmp, q_ct1);
    sycl::free(d_BufOut, q_ct1);
    return 0;
}
