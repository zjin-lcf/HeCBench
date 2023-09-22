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
#include <omp.h>

///////////////////////////////////////////////////////////////////////////////
// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
///////////////////////////////////////////////////////////////////////////////
int ViterbiGPU(float &viterbiProb,
    int   *__restrict viterbiPath,
    int   *__restrict obs, 
    const int nObs, 
    float *__restrict initProb,
    float *__restrict mtState, 
    //const int &nState,
    const int nState,
    const int nEmit,
    float *__restrict mtEmit)
{
  float maxProbNew[nState];
  int path[(nObs-1)*nState];
  float *maxProbOld = initProb;

#pragma omp target data map(to:initProb[0:nState], \
                               mtState[0:nState*nState], \
                               mtEmit[0:nEmit*nState], \
                               obs[0:nObs],\
                               maxProbOld[0:nState]) \
                        map(from: maxProbNew[0:nState], path[0:(nObs-1)*nState])
  {
    auto start = std::chrono::steady_clock::now();

    // main iteration of Viterbi algorithm
    for (int t = 1; t < nObs; t++) // for every input observation
    { 
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int iState = 0; iState < nState; iState++) 
      {
        // find the most probable previous state leading to iState
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

      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int iState = 0; iState < nState; iState++) 
      {
        maxProbOld[iState] = maxProbNew[iState];
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Device execution time of Viterbi iterations %f (s)\n", time * 1e-9f);
  }

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
