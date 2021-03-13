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

// *********************************************************************
// A simple demo application that implements a
// vector dot product computation between 2 float arrays. 
//
// Runs computations with on the GPU device and then checks results 
// against basic host CPU/C++ computation.
// *********************************************************************

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include "shrUtils.h"

// Forward Declarations
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);

void dot_product(const float *a, const float *b, float *c, const int n,
                 sycl::nd_item<3> item_ct1) {
  int iGID = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
             item_ct1.get_local_id(2);
  if (iGID < n) {
    int iInOffset = iGID << 2;
    c[iGID] = a[iInOffset] * b[iInOffset] 
      + a[iInOffset + 1] * b[iInOffset + 1]
      + a[iInOffset + 2] * b[iInOffset + 2]
      + a[iInOffset + 3] * b[iInOffset + 3];
  }
}

int main(int argc, char **argv)
{
 dpct::device_ext &dev_ct1 = dpct::get_current_device();
 sycl::queue &q_ct1 = dev_ct1.default_queue();
  int iNumElements = atoi(argv[1]);

  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
  // rounded up to the nearest multiple of the LocalWorkSize
  int szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  

  // Allocate and initialize host arrays
  float* srcA = (float *)malloc(sizeof(float) * szGlobalWorkSize * 4);
  float* srcB = (float *)malloc(sizeof(float) * szGlobalWorkSize * 4);
  float*  dst = (float *)malloc(sizeof(float) * szGlobalWorkSize);
  float* Golden = (float *)malloc(sizeof(float) * iNumElements);
  shrFillArray(srcA, 4 * iNumElements);
  shrFillArray(srcB, 4 * iNumElements);

  float *d_srcA;
  float *d_srcB;
  float *d_dst;

  d_srcA =
      (float *)sycl::malloc_device(sizeof(float) * szGlobalWorkSize * 4, q_ct1);
  q_ct1.memcpy(d_srcA, srcA, sizeof(float) * szGlobalWorkSize * 4).wait();

  d_srcB =
      (float *)sycl::malloc_device(sizeof(float) * szGlobalWorkSize * 4, q_ct1);
  q_ct1.memcpy(d_srcB, srcB, sizeof(float) * szGlobalWorkSize * 4).wait();

  d_dst = sycl::malloc_device<float>(szGlobalWorkSize, q_ct1);

  printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
      szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize));
  sycl::range<3> grid(1, 1,
                      szGlobalWorkSize % szLocalWorkSize +
                          szGlobalWorkSize / szLocalWorkSize);
  sycl::range<3> block(1, 1, szLocalWorkSize);

  for (int i = 0; i < 100; i++)
    /*
    DPCT1049:0: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    q_ct1.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                       [=](sycl::nd_item<3> item_ct1) {
                         dot_product(d_srcA, d_srcB, d_dst, iNumElements,
                                     item_ct1);
                       });
    });

  q_ct1.memcpy(dst, d_dst, sizeof(float) * szGlobalWorkSize).wait();
  sycl::free(d_dst, q_ct1);
  sycl::free(d_srcA, q_ct1);
  sycl::free(d_srcB, q_ct1);

  // Compute and compare results for golden-host and report errors and pass/fail
  printf("Comparing against Host/C++ computation...\n\n"); 
  DotProductHost ((const float*)srcA, (const float*)srcB, (float*)Golden, iNumElements);
  shrBOOL bMatch = shrComparefet((const float*)Golden, (const float*)dst, (unsigned int)iNumElements, 0.0f, 0);
  printf("\nGPU Result %s CPU Result\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

  free(srcA);
  free(srcB);
  free(dst);
  free(Golden);
  return EXIT_SUCCESS;
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements)
{
  int i, j, k;
  for (i = 0, j = 0; i < iNumElements; i++) 
  {
    pfResult[i] = 0.0f;
    for (k = 0; k < 4; k++, j++) 
    {
      pfResult[i] += pfData1[j] * pfData2[j]; 
    } 
  }
}
