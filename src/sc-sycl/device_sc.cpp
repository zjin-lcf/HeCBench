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

// Device auxiliary functions
void reduce(sycl::nd_item<1> &item, int *l_count, int local_cnt, int *l_data) {
  const int tid       = item.get_local_id(0);
  const int localSize = item.get_local_range(0);
  // load shared mem
  l_data[tid] = local_cnt;
  item.barrier(sycl::access::fence_space::local_space);

  // do reduction in shared mem
  for(int s = localSize >> 1; s > 0; s >>= 1) {
    if(tid < s) {
      l_data[tid] += l_data[tid + s];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  // write result for this block to global mem
  if(tid == 0)
    *l_count = l_data[0];
}

int block_binary_prefix_sums(sycl::nd_item<1> &item, int *l_count, int x, int *l_data) {
  const int tid       = item.get_local_id(0);
  const int localSize = item.get_local_range(0);

  l_data[tid] = x;
  const int length = localSize;
  // Build up tree
  int offset = 1;
  for(int l = length >> 1; l > 0; l >>= 1) {
    item.barrier(sycl::access::fence_space::local_space);
    if(tid < l) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      l_data[bi] += l_data[ai];
    }
    offset <<= 1;
  }
  if(offset < length) {
    offset <<= 1;
  }
  // Build down tree
  int maxThread = offset >> 1;
  for(int d = 0; d < maxThread; d <<= 1) {
    d += 1;
    offset >>= 1;
    item.barrier(sycl::access::fence_space::local_space);
    if(tid < d) {
      int ai = offset * (tid + 1) - 1;
      int bi = ai + (offset >> 1);
      l_data[bi] += l_data[ai];
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
  int output = l_data[tid] + *l_count - x;
  item.barrier(sycl::access::fence_space::local_space);
  if(tid == localSize - 1)
    *l_count += l_data[tid];

  return output;
}

void StreamCompaction (sycl::nd_item<1> &item,
                       int *__restrict l_data, 
                       int size, DATA_TYPE value, int n_tasks, float alpha,
                       DATA_TYPE *__restrict output,
                       const DATA_TYPE *__restrict input,
                       int *__restrict flags
#ifdef DYNAMIC_PARTITION
    , int *__restrict worklist
#endif
    ) {

  int localSize = item.get_local_range(0);
  int tid = item.get_local_id(0);

  int* l_count = &l_data[localSize];
#ifdef DYNAMIC_PARTITION
  int* l_tmp = &l_count[1];
#endif

#ifdef DYNAMIC_PARTITION
  Partitioner p = partitioner_create(n_tasks, alpha, worklist, l_tmp);
#else
  Partitioner p = partitioner_create(n_tasks, alpha);
#endif

  for(int my_s = gpu_first(&p, item); gpu_more(&p); my_s = gpu_next(&p, item)) {

    if(tid == 0) {
      l_count[0] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);

    int local_cnt = 0;
    // Declare on-chip memory
    DATA_TYPE reg[REGS];
#ifdef DYNAMIC_PARTITION
    int pos = my_s * REGS * localSize + tid;
#else
    int pos = (my_s - p.cut) * REGS * localSize + tid;
#endif
    // Load in on-chip memory
#pragma unroll
    for(int j = 0; j < REGS; j++) {
      if(pos < size) {
        reg[j] = input[pos];
        if(reg[j] != value)
          local_cnt++;
      } else
        reg[j] = value;
      pos += localSize;
    }
    reduce(item, &l_count[0], local_cnt, &l_data[0]);

    // Set global synch
    if(tid == 0) {
      int p_count;
#ifdef DYNAMIC_PARTITION
      while((p_count = atomicAdd<sycl::memory_scope::system>(flags[my_s], 0)) == 0) {
      }
      atomicAdd<sycl::memory_scope::system>(flags[my_s + 1], p_count + l_count[0]);
#else
      while((p_count = atomicAdd(flags[my_s], 0)) == 0) {
      }
      atomicAdd(flags[my_s + 1], p_count + l_count[0]);
#endif
      l_count[0] = p_count - 1;
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Store to global memory
#pragma unroll
    for(int j = 0; j < REGS; j++) {
      pos = block_binary_prefix_sums(item, &l_count[0], (int)((reg[j] != value) ? 1 : 0), &l_data[0]);
      if(reg[j] != value) {
        output[pos] = reg[j];
      }
    }
  }
}

void call_StreamCompaction_kernel(sycl::queue &q, int blocks, int threads, int size,
                                  DATA_TYPE value, int n_tasks, float alpha,
                                  DATA_TYPE *output, DATA_TYPE *input, int *flags,
                                  int l_mem_size
#ifdef DYNAMIC_PARTITION
                                  , int *worklist
#endif
    ){
  sycl::range<1> gws (threads * blocks);
  sycl::range<1> lws (threads);

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> sm (sycl::range<1>(l_mem_size/sizeof(int)), cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      StreamCompaction(item, sm.get_pointer(), size, value, n_tasks, alpha, output, input, flags
#ifdef DYNAMIC_PARTITION
      , worklist
#endif
      );
    });
  });
}
