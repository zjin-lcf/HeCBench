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
#include "common.h"

// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
int ViterbiGPU(float &viterbiProb,
    int   *viterbiPath,
    int   *obs, 
    const int nObs, 
    float *initProb,
    float *mtState, 
    //const int &nState,
    const int nState,
    const int nEmit,
    float *mtEmit)
{
  float maxProbNew[nState];
  int path[(nObs-1)*nState];

  { // sycl scope
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<float,1> d_mtState(mtState, nState*nState);
  buffer<float,1> d_mtEmit(mtEmit, nEmit*nState);
  buffer<int,1> d_obs(obs, nObs);
  buffer<float,1> d_maxProbOld(nState);
  buffer<float,1> d_maxProbNew(maxProbNew, nState);
  buffer<int,1> d_path(path, (nObs-1)*nState);

  range<1> global_size ((nState + 255)/256*256);
  range<1> local_size (256);

  // initial probability
  q.submit([&] (handler &h) {
    auto maxProbOld = d_maxProbOld.get_access<sycl_write>(h);
    h.copy(initProb, maxProbOld);
  });

  q.wait();
  auto start = std::chrono::steady_clock::now();

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  { 
    q.submit([&] (handler &h) {
      auto maxProbOld = d_maxProbOld.get_access<sycl_read>(h);
      auto mtState = d_mtState.get_access<sycl_read>(h);
      auto mtEmit = d_mtEmit.get_access<sycl_read>(h);
      auto obs = d_obs.get_access<sycl_read>(h);
      auto maxProbNew = d_maxProbNew.get_access<sycl_discard_write>(h);
      auto path = d_path.get_access<sycl_discard_write>(h);
      h.parallel_for<class hmm>(nd_range<1>(global_size, local_size), [=] (nd_item<1> item) {
        // find the most probable previous state leading to iState
        int iState = item.get_global_id(0);
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
      });
    });

    q.submit([&] (handler &h) {
      auto maxProbNew_acc = d_maxProbNew.get_access<sycl_read>(h);
      auto maxProbOld_acc = d_maxProbOld.get_access<sycl_discard_write>(h);
      h.copy(maxProbNew_acc, maxProbOld_acc);
    });
  }
  q.wait();
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
