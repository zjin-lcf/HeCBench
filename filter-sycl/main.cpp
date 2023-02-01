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
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "common.h"

// compare device results with host results
bool check(queue &q, int *d_nres, int *d_output, int h_nres, std::vector<int> &h_output) {
  int nres;
  q.memcpy(&nres, d_nres, sizeof(int)).wait();

  std::vector<int> output (nres);

  q.memcpy(output.data(), d_output, sizeof(int) * nres).wait();

  // clear device output
  q.memset(d_output, 0, sizeof(int) * nres);

  std::sort(output.begin(), output.end());

  bool equal = (h_nres == nres) && 
               std::equal(h_output.begin(),
                          h_output.begin() + h_nres, output.begin());
  return equal;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
    
  std::vector<int> input (num_elems);

  // Generate input data.
  for (int i = 0; i < num_elems; i++) {
    input[i] = i - num_elems / 2;
  }

  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  // Generate host output with host filtering code.
  std::vector<int> h_output (num_elems);

  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      h_output[h_flt_count++] = input[i];
    }
  }
  // Sort the result for comparison
  std::sort(h_output.begin(), h_output.begin() + h_flt_count);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel, property::queue::in_order());

  int *d_input, *d_output, *d_nres;

  d_input = malloc_device<int>(num_elems, q);
  d_output = malloc_device<int>(num_elems, q);
  d_nres = malloc_device<int>(1, q);

  q.memcpy(d_input, input.data(), sizeof(int) * num_elems);

  range<1> lws (block_size);
  range<1> gws ((num_elems + block_size - 1) / block_size * block_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.memset(d_nres, 0, sizeof(int));

    q.submit([&](handler &h) {
      accessor <int, 1, sycl_read_write, access::target::local> l_n (1, h);
      h.parallel_for(nd_range<1>(gws, lws), [=](nd_item<1> item) {

        int i = item.get_global_id(0); 

        // zero the counter
        if (item.get_local_id(0) == 0) l_n[0] = 0;
        item.barrier(access::fence_space::local_space);

        // get the value, evaluate the predicate, and
        // increment the counter if needed
        int d, pos;

        if(i < num_elems) {
          d = d_input[i];
          if(d > 0) {
            auto ao = atomic_ref<int, 
                                 memory_order::relaxed,
                                 memory_scope::work_group,
                                 access::address_space::local_space> (l_n[0]);
            pos = ao.fetch_add(1);
          }
        }
        item.barrier(access::fence_space::local_space);

        // leader increments the global counter
        if(item.get_local_id(0) == 0) {
          auto ao = atomic_ref<int, 
                               memory_order::relaxed,
                               memory_scope::device,
                               access::address_space::global_space> (d_nres[0]);
          l_n[0] = ao.fetch_add(l_n[0]);
        }
        item.barrier(access::fence_space::local_space);

        // threads with true predicates write their elements
        if(i < num_elems && d > 0) {
          pos += l_n[0]; // increment local pos by global counter
          d_output[pos] = d;
        }
        item.barrier(access::fence_space::local_space);

      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of filter (shared memory) %lf (ms)\n",
         (time * 1e-6) / repeat);

  bool match = check(q, d_nres, d_output, h_flt_count, h_output);
  printf("%s\n", match ? "PASS" : "FAIL");

  free(d_input, q);
  free(d_output, q);
  free(d_nres, q);

  return 0;
}
