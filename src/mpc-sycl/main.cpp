/*
MPC code [double] (LnVs BIT LVs ZE): A GPU-based compressor for arrays of 
double-precision floating-point values.  See the following publication for
more information: http://cs.txstate.edu/~mb92/papers/cluster15.pdf.

Copyright (c) 2015-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of Texas State University nor the names of its
     contributors may be used to endorse or promote products derived from
     this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Annie Yang and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/MPC/.

Publication: This work is described in detail in the following paper.
Annie Yang, Hari Mukka, Farbod Hesaaraki, and Martin Burtscher. MPC: A
Massively Parallel Compression Algorithm for Scientific Data. Proceedings
of the IEEE International Conference on Cluster Computing, pp. 381-389.
September 2015.
*/

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cassert>
#include <string>
#include <sys/time.h>
#include "utils.h"

using std::string;

#define TPB 1024  /* do not change */

static inline 
void prefixsum(int &val, int sbuf[TPB], sycl::nd_item<1> &item)
{
  const int warp = item.get_local_id(0) >> 5;
  const int lane = item.get_local_id(0) & 31;

  auto sg = item.get_sub_group();

  for (int d = 1; d < 32; d *= 2) {
    int tmp = sycl::shift_group_right(sg, val, d);
    if (lane >= d) val += tmp;
  }
  if (lane == 31) sbuf[warp] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp == 0) {
    int v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      int tmp = sycl::shift_group_right(sg, v, d);
      if (lane >= d) v += tmp;
    }
    sbuf[lane] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

static inline 
void prefixsumlong(long &val, long sbuf[TPB], sycl::nd_item<1> &item)
{
  const int warp = item.get_local_id(0) >> 5;
  const int lane = item.get_local_id(0) & 31;

  auto sg = item.get_sub_group();

  for (int d = 1; d < 32; d *= 2) {
    unsigned int tmpl = sycl::shift_group_right(sg, (int)val, d);
    long tmph = sycl::shift_group_right(sg, (int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if (lane == 31) sbuf[warp] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp == 0) {
    long v = sbuf[lane];
    for (int d = 1; d < 32; d *= 2) {
      unsigned int tmpl = sycl::shift_group_right(sg, (int)v, d);
      long tmph = sycl::shift_group_right(sg, (int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[lane] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[warp - 1];
  }
}

static inline 
void prefixsumdimlong(long &val, long sbuf[TPB], const unsigned char dim,
                      sycl::nd_item<1> &item)
{
  const int tid = item.get_local_id(0);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int tix = (warp * dim) + (tid % dim);

  auto sg = item.get_sub_group();

  for (int d = dim; d < 32; d *= 2) {
    unsigned int tmpl = sycl::shift_group_right(sg, (int)val, d);
    long tmph = sycl::shift_group_right(sg, (int)(val >> 32), d);
    if (lane >= d) val += (tmph << 32) + tmpl;
  }
  if ((lane + dim) > 31) sbuf[tix] = val;

  item.barrier(sycl::access::fence_space::local_space);

  if (warp < dim) {
    const int idx = (lane * dim) + warp;
    long v = sbuf[idx];
    for (int d = 1; d < 32; d *= 2) {
      unsigned int tmpl = sycl::shift_group_right(sg, (int)v, d);
      long tmph = sycl::shift_group_right(sg, (int)(v >> 32), d);
      if (lane >= d) v += (tmph << 32) + tmpl;
    }
    sbuf[idx] = v;
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (warp > 0) {
    val += sbuf[tix - dim];
  }
}

/*****************************************************************************
This is the GPU compression kernel, which requires 1024 threads per block and
should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
n: the number of double values to be compressed
original: the input array holding the n doubles (has to be cast to a long array)
goffset: a temporary array with m elements where m = number of thread blocks
dim: the dimensionality of the input data (dim must be between 1 and 32)

Output
------
compressed: the output array that holds the compressed data in integer format

The output array needs to provide space for up to 2 + n + (n + 31) / 32 elements.
The upper half of the first element specifies how many elements are actually
used.  It should be replaced by the value n before the data is further processed.
*****************************************************************************/

static 
void MPCcompress(
  const int n, 
  long* __restrict const original,
  long* __restrict const compressed,
  volatile int* __restrict const goffset,
  unsigned char dim,
  sycl::nd_item<1> &item,
  int *start,
  int *top,
  long *sbuf1,
  long *sbuf2)
{
  const int tid = item.get_local_id(0);
  const int tidm1 = tid - 1;
  const int tidmdim = tid - dim;
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = item.get_group(0);
  const int gdim = item.get_group_range(0);
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;
  auto sg = item.get_sub_group();

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    long v1 = 0;
    if (idx < n) {
      v1 = original[idx];
      sbuf1[tid] = v1;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid >= dim) {
      if (idx < n) {
        v1 -= sbuf1[tidmdim];
      }
    }
    sbuf2[tid] = v1;

    item.barrier(sycl::access::fence_space::local_space);

    long v2 = 0;

    for (int i = 63; i >= 0; i--) {
      v2 = (v2 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }
    sbuf1[tid] = v2;

    item.barrier(sycl::access::fence_space::local_space);

    if (tid > 0) {
      v2 -= sbuf1[tidm1];
    }

    int loc = 0;
    if (v2 != 0) loc = 1;

    unsigned int bitmap = sycl::reduce_over_group(
        sg, loc ? (0x1 << sg.get_local_linear_id()) : 0,
        sycl::ext::oneapi::plus<>());

    if (lanex == 32) {
      sbuf2[tid] = bitmap;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (lanex == 0) {
      if (idx < n) compressed[1 + idx / 64] = (sbuf2[tid + 32] << 32) + bitmap;
    }

    prefixsum(loc, (int *)sbuf1, item);

    if (v2 != 0) {
      sbuf2[loc - 1] = v2;
    }

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc;
      goffset[bid] = -1;
      if (chunk == chunksm1) {
        compressed[0] = (((long)(st + loc)) << 32) + (0x43504d00 - 1) + dim;
      }
      *top = loc;
      *start = st;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid < *top) {
      compressed[*start + tid] = sbuf2[tid];
    }
  }
}

/*****************************************************************************
This is the GPU decompression kernel, which requires 1024 threads per block
and should be launched with as many blocks as the GPU can run simultaneously.

Inputs
------
compressed: the input array holding the compressed data in integer format
goffset: a temporary array with m elements where m = number of thread blocks

The upper half of the first element must hold the value n, i.e., the number
of doubles that the data will generate upon decompression.

Output
------
decompressed: the output array holding the decompressed data in integer format

The output array needs to provide space for n elements has to be cast to an
array of doubles before it can be used.
*****************************************************************************/

static 
void MPCdecompress(
  long* __restrict const compressed, 
  long* __restrict const decompressed,
  volatile int* __restrict const goffset,
  sycl::nd_item<1> &item,
  int *__restrict start,
  int *__restrict top,
  long *__restrict sbuf1,
  long *__restrict sbuf2)
{
  const int dim = (compressed[0] & 31) + 1;
  const int n = compressed[0] >> 32;
  const int tid = item.get_local_id(0);
  const int lanex = tid & 63;
  const int warpx = tid & 0x3c0;
  const int bid = item.get_group(0);
  const int gdim = item.get_group_range(0);
  const int bid1 = ((bid + 1) == gdim) ? 0 : (bid + 1);
  const int init = 1 + (n + 63) / 64;
  const int nru = (n - 1) | 63;
  const int chunksm1 = ((n + (TPB - 1)) / TPB) - 1;

  for (int chunk = bid; chunk <= chunksm1; chunk += gdim) {
    const int idx = tid + chunk * TPB;

    int flag = 0;
    if (idx <= nru) {
      flag = (compressed[1 + idx / 64] >> lanex) & 1;
    }
    int loc = flag;

    item.barrier(sycl::access::fence_space::local_space);

    prefixsum(loc, (int *)sbuf1, item);

    if (tid == (TPB - 1)) {
      int st = init;
      if (chunk > 0) {
        do {
          st = goffset[bid];
        } while (st < 0);  // busy waiting
      }
      goffset[bid1] = st + loc;
      goffset[bid] = -1;
      *top = loc;
      *start = st;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (tid < *top) {
      sbuf2[tid] = compressed[*start + tid];
    }

    item.barrier(sycl::access::fence_space::local_space);

    long v2 = 0;
    if (flag != 0) {
      v2 = sbuf2[loc - 1];
    }

    prefixsumlong(v2, sbuf1, item);

    sbuf2[tid] = v2;

    item.barrier(sycl::access::fence_space::local_space);

    long v1 = 0;
    for (int i = 63; i >= 0; i--) {
      v1 = (v1 << 1) + ((sbuf2[warpx + i] >> lanex) & 1);
    }

    prefixsumdimlong(v1, sbuf1, dim, item);

    if (idx < n) {
      decompressed[idx] = v1;
    }
  }
}

int main(int argc, char *argv[])
{
  printf("MPC - Massively Parallel Compression [double] (%s)\n", __FILE__);
  printf("Copyright 2015-2020 Texas State University\n\n");

  assert(sizeof(long) == sizeof(double));

  if ((argc != 2) && (argc != 3)) {
    printf("usage: %s file [dimension]\n", argv[0]);
    exit(-1);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int wgs = q.get_device().get_info<sycl::info::device::max_work_group_size>();
  if (wgs < TPB) {
    printf("The max work group size supported is less than %d required for the program\n", TPB);
    exit(-1);
  }
  const int SMs = q.get_device().get_info<sycl::info::device::max_compute_units>();
  const int blocks = SMs * 2;

  int dim, insize, outsize;

  string name = argv[1];
  long* const input = readFile(name.c_str(), insize);

  if (argc == 3) {
    dim = atoi(argv[2]);
    outsize = insize + 1 + (insize + 63) / 64;
  } else {
    assert(((input[0] >> 8) & 0xffffff) == 0x43504d);
    dim = (input[0] & 31) + 1;
    outsize = input[0] >> 32;
  }
  assert(0 < dim);  assert(dim <= 32);
  long* const output = new long[outsize];

  long* d_in, * d_out;
  int* d_offs;
  d_in = sycl::malloc_device<long>(insize, q);
  d_out = sycl::malloc_device<long>(outsize, q);
  d_offs = sycl::malloc_device<int>(blocks, q);
  q.memcpy(d_in, input, insize * sizeof(long)).wait();

  struct timeval start, end;

  sycl::range<1> gws (blocks * TPB);
  sycl::range<1> lws (TPB);

  if (argc == 3) {
    gettimeofday(&start, NULL);
    q.memset(d_offs, -1, blocks * sizeof(int)).wait();
 
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sm_start(1, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sm_top(1, cgh);
      sycl::accessor<long, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sbuf1(TPB, cgh);
      sycl::accessor<long, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sbuf2(TPB, cgh);

      cgh.parallel_for(sycl::nd_range<1>(gws, lws),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            MPCcompress(insize, d_in, d_out, d_offs, dim, item,
                        sm_start.get_pointer(), sm_top.get_pointer(),
                        sbuf1.get_pointer(),
                        sbuf2.get_pointer());
      });
    }).wait();

    gettimeofday(&end, NULL);

    q.memcpy(output, d_out, sizeof(long)).wait();
    outsize = output[0] >> 32;

    double ctime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("compression time: %.2f ms\n", 1000.0 * ctime);
    printf("compression throughput: %.3f GB/s\n", 0.000000001 * sizeof(long) * insize / ctime);
    printf("compression ratio: %.3f\n\n", 1.0 * insize / outsize);

    q.memcpy(output, d_out, outsize * sizeof(long)).wait();
    output[0] = (((long)insize) << 32) + (0x43504d00 - 1) + dim;

    name += ".mpc";
    
  } else {

    gettimeofday(&start, NULL);
    q.memset(d_offs, -1, blocks * sizeof(int)).wait();
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sm_start(1, cgh);
      sycl::accessor<int, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sm_top(1, cgh);
      sycl::accessor<long, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sbuf1(TPB, cgh);
      sycl::accessor<long, 1, sycl::access_mode::read_write,
                     sycl::access::target::local> sbuf2(TPB, cgh);

      cgh.parallel_for(sycl::nd_range<1>(gws, lws),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
            MPCdecompress(
                d_in, d_out, d_offs, item, sm_start.get_pointer(),
                sm_top.get_pointer(), sbuf1.get_pointer(),
                sbuf2.get_pointer());
      });
    }).wait();

    gettimeofday(&end, NULL);

    q.memcpy(output, d_out, outsize * sizeof(long)).wait();

    double dtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
    printf("decompression time: %.2f ms\n", 1000.0 * dtime);
    printf("decompression throughput: %.3f GB/s\n\n", 0.000000001 * sizeof(long) * outsize / dtime);

    name += ".org";
  }

  writeFile(name.c_str(), output, outsize);

  delete [] output;
  delete [] input;

  sycl::free(d_offs, q);
  sycl::free(d_out, q);
  sycl::free(d_in, q);

  return 0;
}
