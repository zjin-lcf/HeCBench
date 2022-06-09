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

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "common.h"

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int nres;
  int *data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));
  int *filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate input data.
  srand(2);
  for (int i = 0; i < num_elems; i++) {
    data_to_filter[i] = rand() % 20;
  }

  {
#ifdef USE_GPU 
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> d_data_to_filter (data_to_filter, num_elems);
  buffer<int, 1> d_filtered_data (filtered_data, num_elems);
  buffer<int, 1> d_nres (&nres, 1);

  range<1> local_work_size(block_size);
  range<1> global_work_size((num_elems + block_size - 1) / block_size * block_size);

  int n = num_elems;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](handler &h) {
      auto nres = d_nres.get_access<sycl_discard_write>(h);
      h.fill(nres, 0);
    });

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

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  } // sycl scope

  int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * num_elems));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres ? "PASS" : "FAIL");

  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
