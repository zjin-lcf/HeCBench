/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <sycl/sycl.hpp>
#include "support/common.h"

void RANSAC_kernel_block(sycl::nd_item<1> &item, 
                         int *__restrict outlier_block_count,
                         const float *__restrict model_param_local,
                         const flowvector *__restrict flowvectors,
                         int flowvector_count,
                         int max_iter,
                         int error_threshold,
                         float convergence_threshold,
                         int *__restrict g_out_id,
                         int *__restrict model_candidate,
                         int *__restrict outliers_candidate) {

  const int tx         = item.get_local_id(0);
  const int bx         = item.get_group(0);
  const int num_blocks = item.get_group_range(0);
  const int block_dim  = item.get_local_range(0);

  float vx_error, vy_error;
  int   outlier_local_count = 0;

  // Each block performs one iteration
  for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

    // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
    const float *model_param = &model_param_local [4 * loop_count];

    // Wait until CPU computes F-o-F model
    if(tx == 0) {
      outlier_block_count[0] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if(model_param[0] == -2011)
      continue;

    // Reset local outlier counter
    outlier_local_count = 0;

    // Compute number of outliers
    for(int i = tx; i < flowvector_count; i += block_dim) {
      flowvector fvreg = flowvectors[i]; // x, y, vx, vy
      vx_error         = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
                         (int)((fvreg.y - model_param[1]) * model_param[3])) - fvreg.vx;
      vy_error = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
                 (int)((fvreg.x - model_param[0]) * model_param[3])) - fvreg.vy;
      if((sycl::fabs(vx_error) >= error_threshold) || (sycl::fabs(vy_error) >= error_threshold)) {
        outlier_local_count++;
      }
    }

    auto ao = sycl::atomic_ref<int, 
                               sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space> (outlier_block_count[0]);
    ao.fetch_add(outlier_local_count);

    item.barrier(sycl::access::fence_space::local_space);

    if(tx == 0) {
      // Compare to threshold
      if(outlier_block_count[0] < flowvector_count * convergence_threshold) {
        auto ao = sycl::atomic_ref<int, 
                                   sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space> (g_out_id[0]);
        int index                 = ao.fetch_add(1);
        model_candidate[index]    = loop_count;
        outliers_candidate[index] = outlier_block_count[0];
      }
    }
  }
}

void call_RANSAC_kernel_block(sycl::queue &q, int blocks, int threads, float *model_param_local,
    flowvector *flowvectors, int flowvector_count, int max_iter, int error_threshold,
    float convergence_threshold, int *g_out_id, int *model_candidate, int *outliers_candidate, int l_mem_size)
{
  sycl::range<1> gws (threads * blocks);
  sycl::range<1> lws (threads);
  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> sm (sycl::range<1>(l_mem_size), cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      RANSAC_kernel_block(item, sm.get_pointer(), model_param_local, flowvectors, 
                          flowvector_count, max_iter, error_threshold, convergence_threshold,
                          g_out_id, model_candidate, outliers_candidate);
    });
  });
}
