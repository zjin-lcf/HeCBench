/*
GFC code: A GPU-based compressor for arrays of double-precision
floating-point values.

Copyright (c) 2011-2020, Texas State University. All rights reserved.

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

Authors: Molly A. O'Neil and Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/GFC/.

Publication: This work is described in detail in the following paper.
Molly A. O'Neil and Martin Burtscher. Floating-Point Data Compression at 75
Gb/s on a GPU. Proceedings of the Fourth Workshop on General Purpose Processing
Using GPUs, pp. 7:1-7:7. March 2011.
*/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "kernels.h"

static void Compress(sycl::queue &q, int blocks, int warpsperblock, int repeat, int dimensionality) {

  // generate a test file with fixed values
  FILE *fp = fopen("input.bin", "wb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open input file input.bin for write.\n");
  }
  for (int i = 0; i < MAX; i++) {
    double t = i;
    fwrite(&t, 8, 1, fp);
  }
  fclose(fp);

  fp = fopen("input.bin", "rb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open input file input.bin for read.\n");
  }

  // allocate CPU buffers
  ull *cbuf = (ull *)malloc(sizeof(ull) * MAX); // uncompressed data
  if (cbuf == NULL) {
    fprintf(stderr, "cannot allocate cbuf\n");
  }

  int doubles = fread(cbuf, 8, MAX, fp);
  if (doubles != MAX) {
    fprintf(stderr, "Error in reading input.bin. Exit\n");
    if (cbuf != NULL) free(cbuf);
    fclose(fp);
    return ;
  }
  fclose(fp);

  const int num_warps = blocks * warpsperblock;

  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data
  if (dbuf == NULL) {
    fprintf(stderr, "cannot allocate dbuf\n");
  }
  int *cut = (int *)malloc(sizeof(int) * num_warps); // chunk boundaries
  if (cut == NULL) {
    fprintf(stderr, "cannot allocate cut\n");
  }
  int *off = (int *)malloc(sizeof(int) * num_warps); // offset table
  if (off == NULL) {
    fprintf(stderr, "cannot allocate off\n");
  }

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + num_warps - 1) / (num_warps);
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < num_warps; i++) {
    curr += per;
    cut[i] = std::min(curr, doubles);
    if (cut[i] - before > 0) {
      d = cut[i] - before;
    }
    before = cut[i];
  }

  // set the pad values to ensure correct prediction
  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = 0;
    }
  } else {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
    }
  }

  // allocate GPU buffers
  ull *cbufd = sycl::malloc_device<ull>(doubles, q);
  q.memcpy(cbufd, cbuf, doubles * sizeof(ull)); // uncompressed data

  char *dbufd = sycl::malloc_device<char>((doubles + 1) / 2 * 17, q); // compressed data

  int *cutd = sycl::malloc_device<int>(num_warps, q); // chunk boundaries
  q.memcpy(cutd, cut, num_warps * sizeof(int));

  int *offd = sycl::malloc_device<int>(num_warps, q); // offset table

  q.wait();
  auto start = std::chrono::steady_clock::now();

  sycl::range<1> gws (WARPSIZE * num_warps);
  sycl::range<1> lws (WARPSIZE * warpsperblock);

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<int, 1> ibufs (sycl::range<1>(32 * (3 * WARPSIZE / 2)), cgh);
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item)
        [[sycl::reqd_sub_group_size(WARPSIZE)]] {
        CompressionKernel(item, dimensionality, cbufd, dbufd, cutd, offd,
                          ibufs.get_pointer());
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  fprintf(stderr, "Average compression kernel execution time %f (s)\n", (time * 1e-9f) / repeat);

  // transfer offsets back to CPU
  q.memcpy(off, offd, sizeof(int) * num_warps).wait();

  // output header
  fp = fopen("output.bin", "wb");
  if (fp == NULL) {
    fprintf(stderr, "Failed to open output file output.bin.\n");
  }

  int num;
  int doublecnt = doubles-padding;
  num = fwrite(&blocks, 1, 1, fp);
  assert(1 == num);
  num = fwrite(&warpsperblock, 1, 1, fp);
  assert(1 == num);
  num = fwrite(&dimensionality, 1, 1, fp);
  assert(1 == num);
  num = fwrite(&doublecnt, 4, 1, fp);
  assert(1 == num);
  // output offset table
  for(int i = 0; i < num_warps; i++) {
    int start = 0;
    if(i > 0) start = cut[i-1];
    off[i] -= ((start+1)/2*17);
    num = fwrite(&off[i], 4, 1, fp); // chunk's compressed size in bytes
    assert(1 == num);
  }
  // output compressed data by chunk
  for(int i = 0; i < num_warps; i++) {
    int offset, start = 0;
    if(i > 0) start = cut[i-1];
    offset = ((start+1)/2*17);
    // transfer compressed data back to CPU by chunk
    q.memcpy(dbuf + offset, dbufd + offset, sizeof(char) * off[i]).wait();

    num = fwrite(&dbuf[offset], 1, off[i], fp);
    assert(off[i] == num);
  }
  fclose(fp);

  // compression ratio
  fp = fopen("input.bin", "rb");
  fseek (fp, 0, SEEK_END);
  long input_size = ftell (fp);

  fp = fopen("output.bin", "rb");
  fseek (fp, 0, SEEK_END);
  long output_size = ftell (fp);

  fprintf(stderr, "Compression ratio = %lf\n", 1.0 * input_size / output_size);

  sycl::free(cbufd, q);
  sycl::free(dbufd, q);
  sycl::free(cutd, q);
  sycl::free(offd, q);
  free(cbuf);
  free(dbuf);
  free(cut);
  free(off);
}

/************************************************************************************/

static void VerifySystemParameters()
{
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));

  int val = 1;
  assert(1 == *((char *)&val));
   
  if ((WARPSIZE <= 0) || ((WARPSIZE & (WARPSIZE-1)) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }
}

/************************************************************************************/

int main(int argc, char *argv[])
{
  fprintf(stderr, "GPU FP Compressor v2.2\n");
  fprintf(stderr, "Copyright 2011-2020 Texas State University\n");

  VerifySystemParameters();

  int blocks, warpsperblock, dimensionality;
  int repeat;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  if((4 == argc) || (5 == argc)) { /* compress */
    blocks = atoi(argv[1]);
    assert((0 < blocks) && (blocks < 256));

    warpsperblock = atoi(argv[2]);
    assert((0 < warpsperblock) && (warpsperblock < 256));

    repeat = atoi(argv[3]);

    if(4 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[4]);
    }
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));

    Compress(q, blocks, warpsperblock, repeat, dimensionality);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s <blocks> <warps/block> <repeat> <dimensionality>\n", argv[0]);
    fprintf(stderr, "\ninput.bin is generated by the program and the compressed output file is output.bin.\n");
  }

  return 0;
}
