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

#define _GPU_COMPILER_

#include "support/common.h"
#include "support/partitioner.h"

void Padding_kernel(int n, int m, int pad, int n_tasks, float alpha,
                    DATA_TYPE *__restrict matrix_out,
                    const DATA_TYPE *matrix,
                    int *__restrict flags
#ifdef DYNAMIC_PARTITION
                    , int *__restrict worklist
                    , int *__restrict l_tmp
#endif
                    , sycl::nd_item<1> &item
    ) {

#ifdef DYNAMIC_PARTITION
    Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
    Partitioner p = partitioner_create(n_tasks, alpha);
#endif

    int blockDim_x = item.get_local_range(0);
    int threadIdx_x = item.get_local_id(0);

    const int matrix_size = m * (n + pad);
    const int matrix_size_align =
        (matrix_size + blockDim_x * REGS - 1) / (blockDim_x * REGS) * (blockDim_x * REGS);

    for(int my_s = gpu_first(&p, item); gpu_more(&p); my_s = gpu_next(&p, item)) {

        // Declare on-chip memory
        DATA_TYPE   reg[REGS];
        int pos      = matrix_size_align - 1 - (my_s * REGS * blockDim_x + threadIdx_x);
        int my_s_row = pos / (n + pad);
        int my_x     = pos % (n + pad);
        int pos2     = my_s_row * n + my_x;
// Load in on-chip memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos2 >= 0 && my_x < n && pos2 < matrix_size)
                reg[j] = matrix[pos2];
            else
                reg[j] = 0;
            pos -= blockDim_x;
            my_s_row = pos / (n + pad);
            my_x     = pos % (n + pad);
            pos2     = my_s_row * n + my_x;
        }

        item.barrier(sycl::access::fence_space::local_space);

        // Set global synch
        if(threadIdx_x == 0) {
#ifdef DYNAMIC_PARTITION
            while(atomicAdd<sycl::memory_scope::system>(flags[my_s], 0) == 0) {
            }
            atomicAdd<sycl::memory_scope::system>(flags[my_s + 1], 1);
#else
            while(atomicAdd(flags[my_s], 0) == 0) {
            }
            atomicAdd(flags[my_s + 1], 1);
#endif
        }
        item.barrier(sycl::access::fence_space::local_space);

        pos = matrix_size_align - 1 - (my_s * REGS * blockDim_x + threadIdx_x);
// Store to global memory
#pragma unroll
        for(int j = 0; j < REGS; j++) {
            if(pos >= 0 && pos < matrix_size)
                matrix_out[pos] = reg[j];
            pos -= blockDim_x;
        }
    }
}

void call_Padding_kernel(sycl::queue &q, int blocks, int threads, int n, int m,
                         int pad, int n_tasks, float alpha, 
                         DATA_TYPE *matrix_out, DATA_TYPE *matrix, int *flags
#ifdef DYNAMIC_PARTITION
                         , int l_mem_size, int *worklist
#endif
    ){
  sycl::range<1> gws (threads * blocks);
  sycl::range<1> lws (threads);

  q.submit([&](sycl::handler &cgh) {
#ifdef DYNAMIC_PARTITION
      sycl::local_accessor<int, 1> sm (sycl::range<1>(l_mem_size), cgh);
#endif

    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {

    Padding_kernel(n, m, pad, n_tasks, alpha, matrix_out, matrix, flags
#ifdef DYNAMIC_PARTITION
        , worklist, sm.get_pointer()
#endif
        , item);
    });
  });
}
