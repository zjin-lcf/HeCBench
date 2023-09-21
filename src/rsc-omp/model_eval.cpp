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

#include <omp.h>
#include <math.h>
#include "support/common.h"

void call_RANSAC_kernel_block(int blocks, int threads, float *model_param_local,
    flowvector *flowvectors, int flowvector_count, int max_iter, int error_threshold,
    float convergence_threshold, int *g_out_id, int *model_candidate, int *outliers_candidate)
{
  #pragma omp target teams num_teams(blocks) thread_limit(threads)
  {
    int outlier_block_count;
    #pragma omp parallel 
    {
      const int tx         = omp_get_thread_num();
      const int bx         = omp_get_team_num();
      const int num_blocks = omp_get_num_teams();
      const int block_dim  = omp_get_num_threads();

      float vx_error, vy_error;
      int   outlier_local_count = 0;

      // Each block performs one iteration
      for(int loop_count = bx; loop_count < max_iter; loop_count += num_blocks) {

        // xc=model_param_sh[0], yc=model_param_sh[1], D=model_param_sh[2], R=model_param_sh[3]
        const float *model_param = &model_param_local [4 * loop_count];

        // Wait until CPU computes F-o-F model
        if(tx == 0) {
          outlier_block_count = 0;
        }
        #pragma omp barrier

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
          if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
            outlier_local_count++;
          }
        }

        #pragma omp atomic update
        outlier_block_count += outlier_local_count;

        #pragma omp barrier

        if(tx == 0) {
          // Compare to threshold
          if(outlier_block_count < flowvector_count * convergence_threshold) {
            int index;
            #pragma omp atomic capture
            index = g_out_id[0]++;
            model_candidate[index]    = loop_count;
            outliers_candidate[index] = outlier_block_count;
          }
        }
      }
    }
  }
}
