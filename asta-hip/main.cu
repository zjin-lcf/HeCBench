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
#include <hip/hip_runtime.h>

#include "support/common.h"
#include "support/verify.h"

// GPU kernel 
__global__ void PTTWAC_soa_asta(const int A, 
    const int B, 
    const int b, 
    T *input, 
    int *finished, 
    int *head) 
{
  __shared__ int l_mem[2];
  int* done = l_mem;
  int* gid_ = &done[1];

  const int tid = threadIdx.x;
  int       m   = A * B - 1;

  if(tid == 0) // Dynamic fetch
    gid_[0] = atomicAdd(&head[0], 1);
  __syncthreads();

  while(gid_[0] < m) {
    int next_in_cycle = (gid_[0] * A) - m * (gid_[0] / B);
    if(next_in_cycle == gid_[0]) {
      if(tid == 0) // Dynamic fetch
        gid_[0] = atomicAdd(&head[0], 1);
      __syncthreads();
      continue;
    }
    T   data1, data2, data3, data4;
    int i = tid;
    if(i < b)
      data1 = input[gid_[0] * b + i];
    i += blockDim.x;
    if(i < b)
      data2 = input[gid_[0] * b + i];
    i += blockDim.x;
    if(i < b)
      data3 = input[gid_[0] * b + i];
    i += blockDim.x;
    if(i < b)
      data4 = input[gid_[0] * b + i];

    if(tid == 0) {
      //make sure the read is not cached
      done[0] = atomicAdd(&finished[gid_[0]], 0);
    }
    __syncthreads();

    for(; done[0] == 0; next_in_cycle = (next_in_cycle * A) - m * (next_in_cycle / B)) {
      T backup1, backup2, backup3, backup4;
      i = tid;
      if(i < b)
        backup1 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup2 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup3 = input[next_in_cycle * b + i];
      i += blockDim.x;
      if(i < b)
        backup4 = input[next_in_cycle * b + i];

      if(tid == 0) {
        done[0] = atomicExch(&finished[next_in_cycle], (int)1);
      }
      __syncthreads();

      if(!done[0]) {
        i = tid;
        if(i < b)
          input[next_in_cycle * b + i] = data1;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data2;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data3;
        i += blockDim.x;
        if(i < b)
          input[next_in_cycle * b + i] = data4;
      }
      i = tid;
      if(i < b)
        data1 = backup1;
      i += blockDim.x;
      if(i < b)
        data2 = backup2;
      i += blockDim.x;
      if(i < b)
        data3 = backup3;
      i += blockDim.x;
      if(i < b)
        data4 = backup4;
    }

    if(tid == 0) // Dynamic fetch
      gid_[0] = atomicAdd(&head[0], 1);
    __syncthreads();
  }
}


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

  T * d_in_out;
  int * d_finished;
  int * d_head;
  hipMalloc((void**)&d_in_out, in_size * sizeof(T));
  hipMalloc((void**)&d_finished, sizeof(int) * finished_size);
  hipMalloc((void**)&d_head, sizeof(int));
  T *h_in_backup = (T *)malloc(in_size * sizeof(T));

  // Initialize
  const int max_gpu_threads = 256;
  read_input(h_in_out, p);
  memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
  h_head[0].store(0);
  memcpy(h_in_backup, h_in_out, in_size * sizeof(T)); // Backup for reuse across iterations

  // Loop over the CPU or GPU kernel
  for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

    // Reset
    memcpy(h_in_out, h_in_backup, in_size * sizeof(T));
    memset((void *)h_finished, 0, sizeof(std::atomic_int) * finished_size);
    h_head[0].store(0);

    hipMemcpyAsync(d_in_out, h_in_backup, in_size * sizeof(T), hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_finished, h_finished, sizeof(int) * finished_size, hipMemcpyHostToDevice, 0);
    hipMemcpyAsync(d_head, h_head, sizeof(int), hipMemcpyHostToDevice, 0);

    // Launch GPU threads
    assert(p.n_gpu_threads <= max_gpu_threads && 
        "The thread block size is greater than the maximum thread block size that can be used on this device");
    dim3 dimGrid(blocks);
    dim3 dimBlock(threads);
    hipLaunchKernelGGL(PTTWAC_soa_asta, dim3(dimGrid), dim3(dimBlock), 0, 0, p.m, tiled_n, p.s, d_in_out, d_finished, d_head);

  }

  // Copy back
  hipMemcpy(h_in_out, d_in_out, in_size * sizeof(T), hipMemcpyDeviceToHost);

  // Verify
  int status = verify(h_in_out, h_in_backup, tiled_n * p.s, p.m, p.s);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  // Free memory
  free(h_in_out);
  free(h_finished);
  free(h_head);
  free(h_in_backup);

  hipFree(d_in_out);
  hipFree(d_finished);
  hipFree(d_head);

  return 0;
}
