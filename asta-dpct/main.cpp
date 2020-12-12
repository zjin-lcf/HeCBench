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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>  // for_each

#include "support/common.h"
#include "support/verify.h"

// GPU kernel 
void PTTWAC_soa_asta(const int A, 
                                const int B, 
                                const int b, 
                                  T *input, 
                                int *finished, 
                                int *head,
                                sycl::nd_item<3> item_ct1,
                                int *lmem) 
{

    const int tid = item_ct1.get_local_id(2);
  int       m   = A * B - 1;

  if(tid == 0) // Dynamic fetch
        lmem[1] =
            sycl::atomic<int>(sycl::global_ptr<int>(&head[0])).fetch_add(1);
    item_ct1.barrier();

  while(lmem[1] < m) {
    int next_in_cycle = (lmem[1] * A) - m * (lmem[1] / B);
    if(next_in_cycle == lmem[1]) {
      if(tid == 0) // Dynamic fetch
                lmem[1] = sycl::atomic<int>(sycl::global_ptr<int>(&head[0]))
                              .fetch_add(1);
            item_ct1.barrier();
      continue;
    }
    T   data1, data2, data3, data4;
    int i = tid;
    if(i < b)
      data1 = input[lmem[1] * b + i];
        i += item_ct1.get_local_range().get(2);
    if(i < b)
      data2 = input[lmem[1] * b + i];
        i += item_ct1.get_local_range().get(2);
    if(i < b)
      data3 = input[lmem[1] * b + i];
        i += item_ct1.get_local_range().get(2);
    if(i < b)
      data4 = input[lmem[1] * b + i];

    if(tid == 0) {
      //make sure the read is not cached
            lmem[0] =
                sycl::atomic<int>(sycl::global_ptr<int>(&finished[lmem[1]]))
                    .fetch_add(0);
    }
        item_ct1.barrier();

    for(; lmem[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
      T backup1, backup2, backup3, backup4;
      i = tid;
      if(i < b)
        backup1 = input[next_in_cycle * b + i];
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        backup2 = input[next_in_cycle * b + i];
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        backup3 = input[next_in_cycle * b + i];
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        backup4 = input[next_in_cycle * b + i];

      if(tid == 0) {
                lmem[0] =
                    dpct::atomic_exchange(&finished[next_in_cycle], (int)1);
      }
            item_ct1.barrier();

      if(!lmem[0]) {
        i = tid;
        if(i < b)
          input[next_in_cycle * b + i] = data1;
                i += item_ct1.get_local_range().get(2);
        if(i < b)
          input[next_in_cycle * b + i] = data2;
                i += item_ct1.get_local_range().get(2);
        if(i < b)
          input[next_in_cycle * b + i] = data3;
                i += item_ct1.get_local_range().get(2);
        if(i < b)
          input[next_in_cycle * b + i] = data4;
      }
      i = tid;
      if(i < b)
        data1 = backup1;
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        data2 = backup2;
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        data3 = backup3;
            i += item_ct1.get_local_range().get(2);
      if(i < b)
        data4 = backup4;
    }

    if(tid == 0) // Dynamic fetch
            lmem[1] =
                sycl::atomic<int>(sycl::global_ptr<int>(&head[0])).fetch_add(1);
        item_ct1.barrier();
  }
}


// Params ---------------------------------------------------------------------
struct Params {

  int device;
  int n_gpu_threads;
  int n_gpu_blocks;
  int n_warmup;
  int n_reps;
  int m;
  int n;
  int s;

  Params(int argc, char **argv) {
    device        = 0;
    n_gpu_threads = 64;
    n_gpu_blocks  = 16;
    n_warmup      = 10;
    n_reps        = 100;
    m             = 197;
    n             = 35588;
    s             = 32;
    int opt;
    while((opt = getopt(argc, argv, "hd:i:g:w:r:m:n:s:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'i': n_gpu_threads = atoi(optarg); break;
        case 'g': n_gpu_blocks  = atoi(optarg); break;
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
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./trns [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -i <I>    # of device threads per block (default=64)"
        "\n    -g <G>    # of device blocks (default=16)"
        "\n    -w <W>    # of untimed warmup iterations (default=10)"
        "\n    -r <R>    # of timed repetition iterations (default=100)"
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
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();

  const Params p(argc, argv);
  int blocks = p.n_gpu_blocks;
  int threads = p.n_gpu_threads;
  const int max_gpu_threads = 256;
  assert(threads <= max_gpu_threads && 
          "The thread block size is greater than the maximum thread block size that can be used on this device");

  // Allocate
  int tiled_n       = divceil(p.n, p.s);
  int in_size       = p.m * tiled_n * p.s;
  int finished_size = p.m * tiled_n;
  T *h_in_out = (T *)malloc(in_size * sizeof(T));
  int *h_finished = (int *)malloc(sizeof(int) * finished_size);
  int *h_head = (int *)malloc(sizeof(int));

    sycl::range<3> dimGrid(blocks, 1, 1);
    sycl::range<3> dimBlock(threads, 1, 1);

  T * d_in_out;
  int * d_finished;
  int * d_head;
    d_in_out = sycl::malloc_device<double>(in_size, q_ct1);
    d_finished = sycl::malloc_device<int>(finished_size, q_ct1);
    d_head = sycl::malloc_device<int>(1, q_ct1);
  T *h_in_backup = (T *)malloc(in_size * sizeof(T));

  // Initialize
  read_input(h_in_out, p);
  memset((void *)h_finished, 0, sizeof(int) * finished_size);
  h_head[0] = 0;
  memcpy(h_in_backup, h_in_out, in_size * sizeof(T)); // Backup for reuse across iterations

  // Loop over the CPU or GPU kernel
  for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        q_ct1.memcpy(d_in_out, h_in_backup, in_size * sizeof(T));
        q_ct1.memcpy(d_finished, h_finished, sizeof(int) * finished_size);
        q_ct1.memcpy(d_head, h_head, sizeof(int));

        q_ct1.submit([&](sycl::handler &cgh) {
            sycl::accessor<int, 1, sycl::access::mode::read_write,
                           sycl::access::target::local>
                lmem_acc_ct1(sycl::range<1>(2), cgh);

            auto dpct_global_range = dimGrid * dimBlock;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                                 dpct_global_range.get(1),
                                                 dpct_global_range.get(0)),
                                  sycl::range<3>(dimBlock.get(2),
                                                 dimBlock.get(1),
                                                 dimBlock.get(0))),
                [=](sycl::nd_item<3> item_ct1) {
                    PTTWAC_soa_asta(p.m, tiled_n, p.s, d_in_out, d_finished,
                                    d_head, item_ct1,
                                    lmem_acc_ct1.get_pointer());
                });
        });

        q_ct1.memcpy(h_in_out, d_in_out, in_size * sizeof(T));
  }
    dev_ct1.queues_wait_and_throw();

  // Verify answer
  int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);

  // Free memory
  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

    sycl::free(d_in_out, q_ct1);
    sycl::free(d_finished, q_ct1);
    sycl::free(d_head, q_ct1);

  if (status == 0) printf("Test Passed\n");
  return 0;
}
