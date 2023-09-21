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

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void
viterbi (const float*__restrict__ maxProbOld, 
         const float*__restrict__ mtState, 
         const float*__restrict__ mtEmit, 
                 int*__restrict__ obs, 
               float*__restrict__ maxProbNew, 
                 int*__restrict__ path, 
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

// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
int ViterbiGPU(float &viterbiProb,
    int   *viterbiPath,
    int   *obs, 
    const int nObs, 
    float *initProb,
    float *mtState, 
    const int nState,
    const int nEmit,
    float *mtEmit)
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

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  { 
    hipLaunchKernelGGL(viterbi, gridDim, blockDim, 0, 0, d_maxProbOld, d_mtState, d_mtEmit, d_obs, d_maxProbNew, d_path, nState, t);
    hipMemcpy(d_maxProbOld, d_maxProbNew, sizeof(float)*nState, hipMemcpyDeviceToDevice);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time of Viterbi iterations %f (s)\n", time * 1e-9f);

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

  hipFree(d_mtState);
  hipFree(d_mtEmit);
  hipFree(d_obs);
  hipFree(d_maxProbOld);
  hipFree(d_maxProbNew);
  hipFree(d_path);

  return 1;
}
