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
#include "shrUtils.h"

// Forward Declarations
void DotProductHost(const float* pfData1, const float* pfData2, float* pfResult, int iNumElements);

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

  printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
           szGlobalWorkSize, szLocalWorkSize, (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 

  #pragma omp target data map(to: srcA[0:szGlobalWorkSize * 4], srcB[0:szGlobalWorkSize * 4]) \
                          map(from: dst[0:szGlobalWorkSize])
  {
    for (int i = 0; i < 100; i++) {
      #pragma omp target teams distribute parallel for thread_limit(szLocalWorkSize)
      for (int iGID = 0; iGID < iNumElements; iGID++) {
        int iInOffset = iGID << 2;
        dst[iGID] =  srcA[iInOffset]     * srcB[iInOffset] 
                   + srcA[iInOffset + 1] * srcB[iInOffset + 1]
                   + srcA[iInOffset + 2] * srcB[iInOffset + 2]
                   + srcA[iInOffset + 3] * srcB[iInOffset + 3];
      }
    }
  }

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
