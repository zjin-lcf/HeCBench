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
#include <hip/hip_runtime.h>
#include "shrUtils.h"

// Forward Declarations
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);

__global__ void dot_product(const float *a, const float *b, float *c, const int n) {
  int iGID = blockIdx.x * blockDim.x + threadIdx.x;
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

  hipMalloc((void**)&d_srcA, sizeof(float) * szGlobalWorkSize * 4);
  hipMemcpy(d_srcA, srcA, sizeof(float) * szGlobalWorkSize * 4, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_srcB, sizeof(float) * szGlobalWorkSize * 4);
  hipMemcpy(d_srcB, srcB, sizeof(float) * szGlobalWorkSize * 4, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_dst, sizeof(float) * szGlobalWorkSize);

  printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
      szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 
  dim3 grid (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize); 
  dim3 block (szLocalWorkSize);

  for (int i = 0; i < 100; i++) 
    hipLaunchKernelGGL(dot_product, dim3(grid), dim3(block), 0, 0, d_srcA, d_srcB, d_dst, iNumElements);

  hipMemcpy(dst, d_dst, sizeof(float) * szGlobalWorkSize, hipMemcpyDeviceToHost);
  hipFree(d_dst);
  hipFree(d_srcA);
  hipFree(d_srcB);

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
