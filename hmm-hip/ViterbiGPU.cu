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

#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void
viterbi (const float* maxProbOld, 
         const float* mtState, 
         const float* mtEmit, 
         int *obs, 
         float* maxProbNew, 
         int* path, 
         const int nState,
         const int t)
{
  // find the most probable previous state leading to iState
  int iState = blockDim.x * blockIdx.x + threadIdx.x;
  if (iState < nState) {
    float maxProb = 0.0;
    int maxState = -1;
    for (int preState = 0; preState < nState; preState++) 
    {
      float p = maxProbOld[preState] + mtState[iState*nState + preState];
      if (p > maxProb) 
      {
        maxProb = p;
        maxState = preState;
      }
    }
    maxProbNew[iState] = maxProb + mtEmit[obs[t]*nState+iState];
    path[(t-1)*nState+iState] = maxState;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
///////////////////////////////////////////////////////////////////////////////
int ViterbiGPU(float &viterbiProb,
    int   *__restrict__ viterbiPath,
    int   *__restrict__ obs, 
    const int nObs, 
    float *__restrict__ initProb,
    float *__restrict__ mtState, 
    const int nState,
    const int nEmit,
    float *__restrict__ mtEmit)
{

  float maxProbNew[nState];
  int path[(nObs-1)*nState];

  float *d_mtState, *d_mtEmit, *d_maxProbOld, *d_maxProbNew;
  int *d_obs, *d_path;

  hipMalloc((void**)&d_mtState, sizeof(float)*nState*nState);
  hipMalloc((void**)&d_mtEmit, sizeof(float)*nEmit*nState);
  hipMalloc((void**)&d_obs, sizeof(int)*nObs);
  hipMalloc((void**)&d_maxProbOld, sizeof(float)*nState);
  hipMalloc((void**)&d_maxProbNew, sizeof(float)*nState);
  hipMalloc((void**)&d_path, sizeof(int)*(nObs-1)*nState);

  hipMemcpy(d_mtState, mtState, sizeof(float)*nState*nState, hipMemcpyHostToDevice);
  hipMemcpy(d_mtEmit, mtEmit, sizeof(float)*nEmit*nState, hipMemcpyHostToDevice);
  hipMemcpy(d_obs, obs, sizeof(int)*nObs, hipMemcpyHostToDevice);

  dim3 gridDim ((nState + 255)/256);
  dim3 blockDim (256);

  // initial probability
  hipMemcpy(d_maxProbOld, initProb, sizeof(float)*nState, hipMemcpyHostToDevice);

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  { 
    hipLaunchKernelGGL(viterbi, gridDim, blockDim, 0, 0, 
              d_maxProbOld, d_mtState, d_mtEmit, d_obs, d_maxProbNew, d_path, nState, t);
    hipMemcpy(d_maxProbOld, d_maxProbNew, sizeof(float)*nState, hipMemcpyDeviceToDevice);
  }
  hipMemcpy(maxProbNew, d_maxProbNew, sizeof(float)*nState, hipMemcpyDeviceToHost);
  hipMemcpy(path, d_path, sizeof(int)*(nObs-1)*nState, hipMemcpyDeviceToHost);

  // find the final most probable state
  float maxProb = 0.0;
  int maxState = -1;
  for (int i = 0; i < nState; i++) 
  {
    if (maxProbNew[i] > maxProb) 
    {
      maxProb = maxProbNew[i];
      maxState = i;
    }
  }
  viterbiProb = maxProb;

  // backtrace to find the Viterbi path
  viterbiPath[nObs-1] = maxState;
  for (int t = nObs-2; t >= 0; t--) 
  {
    viterbiPath[t] = path[t*nState+viterbiPath[t+1]];
  }

  return 1;
}
