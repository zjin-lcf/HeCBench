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

#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>  // for_each
#include "common.h"

#include "support/common.h"
#include "support/verify.h"


// Params ---------------------------------------------------------------------
struct Params {

  int device;
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_threads;
  int n_warmup;
  int n_reps;
  int m;
  int n;
  int s;

  Params(int argc, char **argv) {
    device        = 0;
    n_gpu_threads  = 64;
    n_gpu_blocks = 16;
    n_threads     = 0;
    n_warmup      = 10;
    n_reps        = 100;
    m             = 197;
    n             = 35588;
    s             = 32;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:t:w:r:m:n:s:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads  = atoi(optarg); break;
        case 'g': n_gpu_blocks = atoi(optarg); break;
        case 't': n_threads     = atoi(optarg); break;
        case 'w': n_warmup      = atoi(optarg); break;
        case 'r': n_reps        = atoi(optarg); break;
        case 'm': m             = atoi(optarg); break;
        case 'n': n             = atoi(optarg); break;
        case 's': s             = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
    assert(((n_gpu_threads > 0 && n_gpu_blocks > 0) ^ (n_threads > 0))
        && "TRNS only runs on CPU-only or GPU-only: './trns -g 0' or './trns -t 0'");
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./trns [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=64)"
        "\n    -g <G>    # of device blocks (default=16)"
        "\n    -t <T>    # of host threads (default=0)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
        "\n"
        "\nData-partitioning-specific options:"
        "\n    TRNS only supports CPU-only or GPU-only execution"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -m <M>    matrix height (default=197)"
        "\n    -n <N>    matrix width (default=35588)"
        "\n    -s <M>    super-element size (default=32)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(T *x_vector, const Params &p) {
  int tiled_n = divceil(p.n, p.s);
  int in_size = p.m * tiled_n * p.s;
  srand(5432);
  for(int i = 0; i < in_size; i++) {
    x_vector[i] = ((T)(rand() % 100) / 100);
  }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

  const Params p(argc, argv);

  // Allocate
  int tiled_n       = divceil(p.n, p.s);
  int in_size       = p.m * tiled_n * p.s;
  int finished_size = p.m * tiled_n;
  T *h_in_out = (T *)malloc(in_size * sizeof(T));
  std::atomic_int *h_finished =
    (std::atomic_int *)malloc(sizeof(std::atomic_int) * finished_size);
  std::atomic_int *h_head = (std::atomic_int *)malloc(sizeof(std::atomic_int));

  int blocks = p.n_gpu_blocks;
  int threads = p.n_gpu_threads;

  range<1> gws (blocks*threads);
  range<1> lws (threads);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<T, 1> d_in_out(in_size);
  buffer<int, 1> d_finished(finished_size);
  buffer<int, 1> d_head(1);

  T *h_in_backup = (T *)malloc(in_size * sizeof(T));

  // Initialize
  const int max_gpu_threads = 256;
  read_input(h_in_out, p);
  memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
  h_head[0].store(0);
  memcpy(h_in_backup, h_in_out, in_size * sizeof(T)); // Backup for reuse across iterations

  const int A = p.m;
  const int B = tiled_n;
  const int b = p.s;

  // Loop over the CPU or GPU kernel
  for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Reset
    memcpy(h_in_out, h_in_backup, in_size * sizeof(T));
    memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
    h_head[0].store(0);

    q.submit([&] (handler &cgh) {
      auto d_acc = d_in_out.template get_access<sycl_discard_write>(cgh);
      cgh.copy(h_in_backup, d_acc);
    });

    q.submit([&] (handler &cgh) {
      auto d_acc = d_finished.template get_access<sycl_discard_write>(cgh);
      cgh.copy(h_finished, d_acc);
    });

    q.submit([&] (handler &cgh) {
      auto d_acc = d_head.template get_access<sycl_discard_write>(cgh);
      cgh.copy(h_head, d_acc);
    });

    // Launch GPU threads
    assert(p.n_gpu_threads <= max_gpu_threads && 
          "The thread block size is greater than the maximum thread block size that can be used on this device");

    q.submit([&] (handler &cgh) {
      auto input = d_in_out.template get_access<sycl_read_write>(cgh);
      auto finished = d_finished.template get_access<sycl_atomic>(cgh);
      auto head = d_head.template get_access<sycl_atomic>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> lmem(2, cgh);

      cgh.parallel_for<class asta>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {

        const int tid = item.get_local_id(0);
        int       m   = A * B - 1;

        if(tid == 0) // Dynamic fetch
          lmem[1] = atomic_fetch_add(head[0], 1);
        item.barrier(access::fence_space::local_space);

        while(lmem[1] < m) {
          int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
          if(next_in_cycle == lmem[1]) {
            if(tid == 0) // Dynamic fetch
              lmem[1] = atomic_fetch_add(head[0], 1);
            item.barrier(access::fence_space::local_space);
            continue;
          }
          T   data1, data2, data3, data4;
          int i = tid;
          if(i < b)
            data1 = input[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data2 = input[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data3 = input[lmem[1] * b + i];
          i += item.get_local_range(0);
          if(i < b)
            data4 = input[lmem[1] * b + i];

          if(tid == 0) {
            //make sure the read is not cached
            lmem[0] = atomic_fetch_add(finished[lmem[1]], 0);
          }
          item.barrier(access::fence_space::local_space);

          for(; lmem[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
            T backup1, backup2, backup3, backup4;
            i = tid;
            if(i < b)
              backup1 = input[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup2 = input[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup3 = input[next_in_cycle * b + i];
            i += item.get_local_range(0);
            if(i < b)
              backup4 = input[next_in_cycle * b + i];

            if(tid == 0) {
              lmem[0] = atomic_exchange(finished[next_in_cycle], (int)1);
            }
            item.barrier(access::fence_space::local_space);

            if(!lmem[0]) {
              i = tid;
              if(i < b)
                input[next_in_cycle * b + i] = data1;
              i += item.get_local_range(0);
              if(i < b)
                input[next_in_cycle * b + i] = data2;
              i += item.get_local_range(0);
              if(i < b)
                input[next_in_cycle * b + i] = data3;
              i += item.get_local_range(0);
              if(i < b)
                input[next_in_cycle * b + i] = data4;
            }
            i = tid;
            if(i < b)
              data1 = backup1;
            i += item.get_local_range(0);
            if(i < b)
              data2 = backup2;
            i += item.get_local_range(0);
            if(i < b)
              data3 = backup3;
            i += item.get_local_range(0);
            if(i < b)
              data4 = backup4;
          }

          if(tid == 0) // Dynamic fetch
            lmem[1] = atomic_fetch_add(head[0], 1);
          item.barrier(access::fence_space::local_space);
        }
      });
    });
  } 

  // Copy back
  q.submit([&] (handler &cgh) {
    auto d_acc = d_in_out.template get_access<sycl_read>(cgh);
    cgh.copy(d_acc, h_in_out);
  });

  q.wait();

  // Verify answer
  int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);

  // Free memory
  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

  if (status == 0) printf("Test Passed\n");
  return 0;
}
