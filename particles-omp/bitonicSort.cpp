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

#include <stdio.h>
#include <assert.h>
#include <omp.h>

#define LOCAL_SIZE_LIMIT 512U
#include "bitonicSort_kernels.cpp"

static unsigned int factorRadix2(unsigned int& log2L, unsigned int L){
  if(!L){
    log2L = 0;
    return 0;
  }else{
    for(log2L = 0; (L & 1) == 0; L >>= 1, log2L++);
    return L;
  }
}

void bitonicSort(
    unsigned int *d_dstKey,
    unsigned int *d_dstVal,
    unsigned int *d_srcKey,
    unsigned int *d_srcVal,
    unsigned int batch,
    unsigned int arrayLength,
    unsigned int dir
    )
{
  if(arrayLength < 2) return;

  //Only power-of-two array lengths are supported so far
  unsigned int log2L;
  unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  size_t localWorkSize, globalWorkSize;
  int numTeams, numThreads;

  if(arrayLength <= LOCAL_SIZE_LIMIT)
  {
    assert( (batch * arrayLength) % LOCAL_SIZE_LIMIT == 0 );

    //Launch bitonicSortLocal
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    numTeams = globalWorkSize/localWorkSize;
    numThreads = localWorkSize;

    bitonicSortLocal(
        d_dstKey,  
        d_dstVal,  
        d_srcKey,  
        d_srcVal,  
        arrayLength,
        dir,
        numTeams,
        numThreads);
  }
  else
  {
    //Launch bitonicSortLocal1
    localWorkSize  = LOCAL_SIZE_LIMIT / 2;
    globalWorkSize = batch * arrayLength / 2;
    numTeams = globalWorkSize/localWorkSize;
    numThreads = localWorkSize;

    bitonicSortLocal1(
        d_dstKey,
        d_dstVal,
        d_srcKey,
        d_srcVal,
        numTeams,
        numThreads);

    for(unsigned int size = 2 * LOCAL_SIZE_LIMIT; size <= arrayLength; size <<= 1)
    {
      for(unsigned stride = size / 2; stride > 0; stride >>= 1)
      {
        if(stride >= LOCAL_SIZE_LIMIT)
        {
          //Launch bitonicMergeGlobal
          localWorkSize  = LOCAL_SIZE_LIMIT / 4;
          globalWorkSize = batch * arrayLength / 2;
          numTeams = globalWorkSize/localWorkSize;
          numThreads = localWorkSize;

          bitonicMergeGlobal(
              d_dstKey,
              d_dstVal,
              d_dstKey,
              d_dstVal,
              arrayLength,
              size,
              stride,
              dir,
              numTeams,
              numThreads);
        }
        else
        {
          //Launch bitonicMergeLocal
          localWorkSize  = LOCAL_SIZE_LIMIT / 2;
          globalWorkSize = batch * arrayLength / 2;
          numTeams = globalWorkSize/localWorkSize;
          numThreads = localWorkSize;

          assert(stride < LOCAL_SIZE_LIMIT);
          bitonicMergeLocal(
              d_dstKey,
              d_dstVal,
              d_dstKey,
              d_dstVal,
              arrayLength,
              size,
              stride,
              dir,
              numTeams,
              numThreads);
          break;
        }
      }
    }
  }
}
