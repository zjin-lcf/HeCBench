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

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"
#include "shrUtils.h"

// Forward Declarations
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int iNumElements = atoi(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  // set and log Global and Local work size dimensions
  const int szLocalWorkSize = 256;

  // rounded up to the nearest multiple of the LocalWorkSize
  const int szGlobalWorkSize = shrRoundUp((int)szLocalWorkSize, iNumElements);  

  const size_t src_size = szGlobalWorkSize * 4;
  const size_t src_size_bytes = src_size * sizeof(float);

  const size_t dst_size = szGlobalWorkSize;
  const size_t dst_size_bytes = dst_size * sizeof(float);

  // Allocate and initialize host arrays
  float* srcA = (float*) malloc (src_size_bytes);
  float* srcB = (float*) malloc (src_size_bytes);
  float*  dst = (float*) malloc (dst_size_bytes);
  float* Golden = (float*) malloc (sizeof(float) * iNumElements);
  shrFillArray(srcA, 4 * iNumElements);
  shrFillArray(srcB, 4 * iNumElements);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  float *d_srcA = malloc_device<float>(src_size, q);
  q.memcpy(d_srcA, srcA, src_size_bytes);

  float *d_srcB = malloc_device<float>(src_size, q);
  q.memcpy(d_srcB, srcB, src_size_bytes);

  float *d_dst = malloc_device<float>(dst_size, q);

  printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 
  range<1> gws (szGlobalWorkSize);
  range<1> lws (szLocalWorkSize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iNumIterations; i++) {
    q.submit([&] (handler &cgh) {
      cgh.parallel_for<class dot_product>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        int iGID = item.get_global_id(0);
        if (iGID < iNumElements) {
          int iInOffset = iGID << 2;
          d_dst[iGID] = d_srcA[iInOffset    ] * d_srcB[iInOffset    ] +
                        d_srcA[iInOffset + 1] * d_srcB[iInOffset + 1] +
                        d_srcA[iInOffset + 2] * d_srcB[iInOffset + 2] +
                        d_srcA[iInOffset + 3] * d_srcB[iInOffset + 3];
        }
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iNumIterations);

  q.memcpy(dst, d_dst, dst_size_bytes).wait();

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
