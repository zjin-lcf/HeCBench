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
// vector dot product computation between 2 Type arrays.
//
// Runs computations with on the GPU device and then checks results
// against basic host CPU/C++ computation.
// *********************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include "shrUtils.h"

typedef int Type;

// Forward Declarations
Type DotProductHost(const Type* pfData1, const Type* pfData2, size_t iNumElements);

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t iNumElements = atol(argv[1]);
  const int iNumIterations = atoi(argv[2]);

  // set and log Global and Local work size dimensions
  int szLocalWorkSize = 256;
  size_t szGlobalWorkSize = shrRoundUp(szLocalWorkSize, iNumElements);

  printf("Global Work Size \t\t= %zu\nLocal Work Size \t\t= %d\n",
         szGlobalWorkSize, szLocalWorkSize);

  const size_t src_size = szGlobalWorkSize;
  const size_t src_size_bytes = src_size * sizeof(Type);

  // Allocate and initialize host arrays
  Type* srcA = (Type*) malloc (src_size_bytes);
  Type* srcB = (Type*) malloc (src_size_bytes);

  size_t i;
  srand(123);
  for (i = 0; i < iNumElements ; ++i)
  {
    srcA[i] = 1;
    srcB[i] = -1;
  }
  for (i = iNumElements; i < src_size ; ++i) {
    srcA[i] = srcB[i] = 0;
  }

  Type dst;
  #pragma omp target data map(to: srcA[0:src_size], srcB[0:src_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < iNumIterations; i++) {
      dst = 0;
      #pragma omp target teams distribute parallel for map(tofrom: dst) \
       reduction(+:dst) thread_limit(szLocalWorkSize)
      for (size_t iGID = 0; iGID < src_size / 4; iGID++) {
        size_t iInOffset = iGID * 4;
        dst += srcA[iInOffset    ] * srcB[iInOffset    ] +
               srcA[iInOffset + 1] * srcB[iInOffset + 1] +
               srcA[iInOffset + 2] * srcB[iInOffset + 2] +
               srcA[iInOffset + 3] * srcB[iInOffset + 3];
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / iNumIterations);
  }

  // Compute and compare results for golden-host and report errors and pass/fail
  printf("Comparing against Host/C++ computation...\n\n");
  Type Golden = DotProductHost (srcA, srcB, iNumElements);
  bool bMatch = std::abs(Golden - dst) < 1e-3f;
  printf("\nGPU Result %s CPU Result\n", bMatch ? "matches" : "DOESN'T match");

  free(srcA);
  free(srcB);
  return EXIT_SUCCESS;
}

// "Golden" Host processing dot product function for comparison purposes
// *********************************************************************
Type DotProductHost(const Type* pfData1, const Type* pfData2, size_t iNumElements)
{
  Type r = 0;
  for (size_t i = 0; i < iNumElements; i++)
  {
    r += pfData1[i] * pfData2[i];
  }
  return r;
}
