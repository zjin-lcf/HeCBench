/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cstdio>
#include "common.h"

#define NUM_ELEMS 10000000
#define NUM_THREADS_PER_BLOCK 256



int main(int argc, char **argv) {
  int nres = 0;
  int *data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));
  int *filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  // Generate input data.
  srand(2);
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % 20;
  }

  {
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> d_data_to_filter (data_to_filter, NUM_ELEMS);
  buffer<int, 1> d_filtered_data (filtered_data, NUM_ELEMS);
  buffer<int, 1> d_nres (&nres, 1);

  range<1> local_work_size(NUM_THREADS_PER_BLOCK);
  range<1> global_work_size((NUM_ELEMS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK * NUM_THREADS_PER_BLOCK);

  int n = NUM_ELEMS;

  q.submit([&](handler &h) {
    accessor <int, 1, sycl_atomic, access::target::local> l_n (1, h);
    auto src = d_data_to_filter.get_access<sycl_read>(h);
    auto dst = d_filtered_data.get_access<sycl_discard_write>(h);
    auto nres = d_nres.get_access<sycl_atomic>(h);
    h.parallel_for(nd_range<1>(global_work_size, local_work_size), [=](nd_item<1> item) {

    int i = item.get_global_id(0); 

    // zero the counter
    if (item.get_local_id(0) == 0)
      l_n[0].store(0);
    item.barrier(access::fence_space::local_space);

    // get the value, evaluate the predicate, and
    // increment the counter if needed
    int d, pos;

    if(i < n) {
      d = src[i];
      if(d > 0)
        pos = atomic_fetch_add(l_n[0], 1);
    }
    item.barrier(access::fence_space::local_space);

    // leader increments the global counter
    if(item.get_local_id(0) == 0) {
      l_n[0].store(atomic_fetch_add(nres[0], atomic_load(l_n[0])));
    }
    item.barrier(access::fence_space::local_space);

    // threads with true predicates write their elements
    if(i < n && d > 0) {
      pos += atomic_load(l_n[0]); // increment local pos by global counter
      dst[pos] = d;
    }
    item.barrier(access::fence_space::local_space);

    });
  });

  }

  int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres ? "PASSED" : "FAILED");

  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
