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
#include <sycl/sycl.hpp>

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_mtState = sycl::malloc_device<float>(nState*nState, q);
  q.memcpy(d_mtState, mtState, sizeof(float)*nState*nState);

  float *d_mtEmit = sycl::malloc_device<float>(nEmit*nState, q);
  q.memcpy(d_mtEmit, mtEmit, sizeof(float)*nEmit*nState);

  int *d_obs = sycl::malloc_device<int>(nObs, q);
  q.memcpy(d_obs, obs, sizeof(int)*nObs);

  float *d_maxProbOld = sycl::malloc_device<float>(nState, q);
  float *d_maxProbNew = sycl::malloc_device<float>(nState, q);
  int *d_path = sycl::malloc_device<int>((nObs-1)*nState, q);

  sycl::range<1> gws ((nState + 255)/256*256);
  sycl::range<1> lws (256);

  // initial probability
  q.memcpy(d_maxProbOld, initProb, sizeof(float)*nState).wait();

  auto start = std::chrono::steady_clock::now();

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  { 
    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class hmm>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        // find the most probable previous state leading to iState
        int iState = item.get_global_id(0);
        if (iState < nState) {
          float maxProb = 0.0;
          int maxState = -1;
          for (int preState = 0; preState < nState; preState++) 
          {
            float p = d_maxProbOld[preState] + d_mtState[iState*nState + preState];
            if (p > maxProb) 
            {
              maxProb = p;
              maxState = preState;
            }
          }
          d_maxProbNew[iState] = maxProb + d_mtEmit[d_obs[t]*nState+iState];
          d_path[(t-1)*nState+iState] = maxState;
        }
      });
    });

    q.memcpy(d_maxProbOld, d_maxProbNew, sizeof(float)*nState);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device execution time of Viterbi iterations %f (s)\n", time * 1e-9f);

  q.memcpy(maxProbNew, d_maxProbNew, sizeof(float)*nState);
  q.memcpy(path, d_path, sizeof(int)*(nObs-1)*nState);

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
  
  sycl::free(d_mtState, q);
  sycl::free(d_mtEmit, q);
  sycl::free(d_obs, q);
  sycl::free(d_maxProbOld, q);
  sycl::free(d_maxProbNew, q);
  sycl::free(d_path, q);

  return 1;
}
