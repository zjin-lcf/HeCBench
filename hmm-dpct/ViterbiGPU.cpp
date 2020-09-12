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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void
viterbi (const float* maxProbOld, 
         const float* mtState, 
         const float* mtEmit, 
         int *obs, 
         float* maxProbNew, 
         int* path, 
         const int nState,
         const int t,
         sycl::nd_item<3> item_ct1)
{
  // find the most probable previous state leading to iState
    int iState = item_ct1.get_local_range().get(2) * item_ct1.get_group(2) +
                 item_ct1.get_local_id(2);
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

  float maxProbNew[nState];
  int path[(nObs-1)*nState];

  float *d_mtState, *d_mtEmit, *d_maxProbOld, *d_maxProbNew;
  int *d_obs, *d_path;

    d_mtState =
        (float *)sycl::malloc_device(sizeof(float) * nState * nState, q_ct1);
    d_mtEmit =
        (float *)sycl::malloc_device(sizeof(float) * nEmit * nState, q_ct1);
    d_obs = sycl::malloc_device<int>(nObs, q_ct1);
    d_maxProbOld = sycl::malloc_device<float>(nState, q_ct1);
    d_maxProbNew = sycl::malloc_device<float>(nState, q_ct1);
    d_path =
        (int *)sycl::malloc_device(sizeof(int) * (nObs - 1) * nState, q_ct1);

    q_ct1.memcpy(d_mtState, mtState, sizeof(float) * nState * nState).wait();
    q_ct1.memcpy(d_mtEmit, mtEmit, sizeof(float) * nEmit * nState).wait();
    q_ct1.memcpy(d_obs, obs, sizeof(int) * nObs).wait();

    sycl::range<3> gridDim((nState + 255) / 256, 1, 1);
    sycl::range<3> blockDim(256, 1, 1);

  // initial probability
    q_ct1.memcpy(d_maxProbOld, initProb, sizeof(float) * nState).wait();

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  {
        q_ct1.submit([&](sycl::handler &cgh) {
            auto dpct_global_range = gridDim * blockDim;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(blockDim.get(2),
                                                 blockDim.get(1),
                                                 blockDim.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    viterbi(d_maxProbOld, d_mtState, d_mtEmit, d_obs,
                            d_maxProbNew, d_path, nState, t, item_ct1);
                });
        });
        q_ct1.memcpy(d_maxProbOld, d_maxProbNew, sizeof(float) * nState).wait();
  }
    q_ct1.memcpy(maxProbNew, d_maxProbNew, sizeof(float) * nState).wait();
    q_ct1.memcpy(path, d_path, sizeof(int) * (nObs - 1) * nState).wait();

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
