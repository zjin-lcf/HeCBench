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

#define DPCT_USM_LEVEL_NONE
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

  float maxProbNew[nState];
  int path[(nObs-1)*nState];

  float *d_mtState, *d_mtEmit, *d_maxProbOld, *d_maxProbNew;
  int *d_obs, *d_path;

    dpct::dpct_malloc((void **)&d_mtState, sizeof(float) * nState * nState);
    dpct::dpct_malloc((void **)&d_mtEmit, sizeof(float) * nEmit * nState);
    dpct::dpct_malloc((void **)&d_obs, sizeof(int) * nObs);
    dpct::dpct_malloc((void **)&d_maxProbOld, sizeof(float) * nState);
    dpct::dpct_malloc((void **)&d_maxProbNew, sizeof(float) * nState);
    dpct::dpct_malloc((void **)&d_path, sizeof(int) * (nObs - 1) * nState);

    dpct::dpct_memcpy(d_mtState, mtState, sizeof(float) * nState * nState,
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_mtEmit, mtEmit, sizeof(float) * nEmit * nState,
                      dpct::host_to_device);
    dpct::dpct_memcpy(d_obs, obs, sizeof(int) * nObs, dpct::host_to_device);

    sycl::range<3> gridDim((nState + 255) / 256, 1, 1);
    sycl::range<3> blockDim(256, 1, 1);

  // initial probability
    dpct::dpct_memcpy(d_maxProbOld, initProb, sizeof(float) * nState,
                      dpct::host_to_device);

  // main iteration of Viterbi algorithm
  for (int t = 1; t < nObs; t++) // for every input observation
  {
        {
            dpct::buffer_t d_maxProbOld_buf_ct0 =
                dpct::get_buffer(d_maxProbOld);
            dpct::buffer_t d_mtState_buf_ct1 = dpct::get_buffer(d_mtState);
            dpct::buffer_t d_mtEmit_buf_ct2 = dpct::get_buffer(d_mtEmit);
            dpct::buffer_t d_obs_buf_ct3 = dpct::get_buffer(d_obs);
            dpct::buffer_t d_maxProbNew_buf_ct4 =
                dpct::get_buffer(d_maxProbNew);
            dpct::buffer_t d_path_buf_ct5 = dpct::get_buffer(d_path);
            dpct::get_default_queue().submit([&](sycl::handler &cgh) {
                auto d_maxProbOld_acc_ct0 =
                    d_maxProbOld_buf_ct0
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto d_mtState_acc_ct1 =
                    d_mtState_buf_ct1
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto d_mtEmit_acc_ct2 =
                    d_mtEmit_buf_ct2.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_obs_acc_ct3 =
                    d_obs_buf_ct3.get_access<sycl::access::mode::read_write>(
                        cgh);
                auto d_maxProbNew_acc_ct4 =
                    d_maxProbNew_buf_ct4
                        .get_access<sycl::access::mode::read_write>(cgh);
                auto d_path_acc_ct5 =
                    d_path_buf_ct5.get_access<sycl::access::mode::read_write>(
                        cgh);

                auto dpct_global_range = gridDim * blockDim;

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                     dpct_global_range.get(1),
                                                     dpct_global_range.get(0)),
                                      sycl::range<3>(blockDim.get(2),
                                                     blockDim.get(1),
                                                     blockDim.get(0))),
                    [=](sycl::nd_item<3> item_ct1) {
                        viterbi((const float *)(&d_maxProbOld_acc_ct0[0]),
                                (const float *)(&d_mtState_acc_ct1[0]),
                                (const float *)(&d_mtEmit_acc_ct2[0]),
                                (int *)(&d_obs_acc_ct3[0]),
                                (float *)(&d_maxProbNew_acc_ct4[0]),
                                (int *)(&d_path_acc_ct5[0]), nState, t,
                                item_ct1);
                    });
            });
        }
        dpct::dpct_memcpy(d_maxProbOld, d_maxProbNew, sizeof(float) * nState,
                          dpct::device_to_device);
  }
    dpct::dpct_memcpy(maxProbNew, d_maxProbNew, sizeof(float) * nState,
                      dpct::device_to_host);
    dpct::dpct_memcpy(path, d_path, sizeof(int) * (nObs - 1) * nState,
                      dpct::device_to_host);

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
