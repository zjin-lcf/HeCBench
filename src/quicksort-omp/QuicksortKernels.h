/*
   Copyright (c) 2014-2019, Intel Corporation
   Redistribution and use in source and binary forms, with or without 
   modification, are permitted provided that the following conditions 
   are met:
 * Redistributions of source code must retain the above copyright 
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above 
 copyright notice, this list of conditions and the following 
 disclaimer in the documentation and/or other materials provided 
 with the distribution.
 * Neither the name of Intel Corporation nor the names of its 
 contributors may be used to endorse or promote products 
 derived from this software without specific prior written 
 permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef QUICKSORT_KNL_H
#define QUICKSORT_KNL_H

#include "Quicksort.h"

#pragma omp declare target
template <typename T>
void plus_prescan( T *a,  T *b) {
  T av = *a;
  T bv = *b;
  *a = bv;
  *b = bv + av;
}

/// bitonic_sort: sort 2*LOCAL_THREADCOUNT elements
  template <typename T>
void bitonic_sort( T* sh_data, const uint localid)
{
  for (uint ulevel = 1; ulevel < LQSORT_LOCAL_WORKGROUP_SIZE; ulevel <<= 1) {
    for (uint j = ulevel; j > 0; j >>= 1) {
      uint pos = 2*localid - (localid & (j - 1));

      uint direction = localid & ulevel;
      uint av = sh_data[pos], bv = sh_data[pos + j];
      const bool sortThem = av > bv;
      const uint greater = Select(bv, av, sortThem);
      const uint lesser  = Select(av, bv, sortThem);

      sh_data[pos]     = Select(lesser, greater, direction);
      sh_data[pos + j] = Select(greater, lesser, direction);
#pragma omp barrier
    }
  }

  for (uint j = LQSORT_LOCAL_WORKGROUP_SIZE; j > 0; j >>= 1) {
    uint pos = 2*localid - (localid & (j - 1));

    uint av = sh_data[pos], bv = sh_data[pos + j];
    const bool sortThem = av > bv;
    sh_data[pos]      = Select(av, bv, sortThem);
    sh_data[pos + j]  = Select(bv, av, sortThem);

#pragma omp barrier
  }
}

  template <typename T>
void sort_threshold( T* data_in, 
    T* data_out,
    uint start, 
    uint end,  
    T* temp, 
    uint localid) 
{
  uint tsum = end - start;
  if (tsum == SORT_THRESHOLD) {
    bitonic_sort(data_in+start, localid);
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      data_out[start + i] = data_in[start + i];
    }
  } else if (tsum > 1) {
    for (uint i = localid; i < SORT_THRESHOLD; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      if (i < tsum) {
        temp[i] = data_in[start + i];
      } else {
        temp[i] = UINT_MAX;
      }
    }
#pragma omp barrier
    bitonic_sort(temp, localid);

    for (uint i = localid; i < tsum; i += LQSORT_LOCAL_WORKGROUP_SIZE) {
      data_out[start + i] = temp[i];
    }
  } else if (tsum == 1 && localid == 0) {
    data_out[start] = data_in[start];
  } 
}
#pragma omp end declare target

// record to push start of the sequence, end of the sequence and direction of sorting on internal stack
typedef struct workstack_record {
  uint start;
  uint end;
  uint direction;
} workstack_record;

#define PUSH(START, END)       if (localid == 0) { \
  ++workstack_pointer; \
  workstack_record wr = { (START), (END), direction ^ 1 }; \
  workstack[workstack_pointer] = wr; \
} 

#endif
