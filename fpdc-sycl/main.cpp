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
#include "common.h"
#include "kernels.h"

static void Compress(queue &q, int blocks, int warpsperblock, int dimensionality) {
  // allocate CPU buffers
  ull *cbuf = (ull *)malloc(sizeof(ull) * MAX); // uncompressed data
  if (cbuf == NULL) {
    fprintf(stderr, "cannot allocate cbuf\n");
  }
  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data
  if (dbuf == NULL) {
    fprintf(stderr, "cannot allocate dbuf\n");
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) {
    fprintf(stderr, "cannot allocate cut\n");
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if (off == NULL) {
    fprintf(stderr, "cannot allocate off\n");
  }

  // read in trace to cbuf
  int doubles = fread(cbuf, 8, MAX, stdin);

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
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
  buffer<ull, 1> cbufd (cbuf, doubles); // uncompressed data
  buffer<char, 1> dbufd ((doubles + 1) / 2 * 17); // compressed data
  buffer<int, 1> cutd (cut, blocks * warpsperblock); // chunk boundaries
  buffer<int, 1> offd (blocks * warpsperblock); // offset table

  range<1> gws (blocks * WARPSIZE * warpsperblock);
  range<1> lws (WARPSIZE * warpsperblock);

  for (int i = 0; i < 100; i++)
    q.submit([&](handler &cgh) {
      auto cbuf = cbufd.get_access<sycl_read>(cgh);
      auto dbuf = dbufd.get_access<sycl_write>(cgh);
      auto cut = cutd.get_access<sycl_read>(cgh);
      auto off = offd.get_access<sycl_write>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> ibufs (2 * (3 * WARPSIZE / 2), cgh);
      cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        CompressionKernel(item, dimensionality,
                          cbuf.get_pointer(), dbuf.get_pointer(), cut.get_pointer(),
                          off.get_pointer(), ibufs.get_pointer());
      });
    });

  // transfer offsets back to CPU
  q.submit([&] (handler &cgh) {
    auto acc = offd.get_access<sycl_read>(cgh);
    cgh.copy(acc, off);
  }).wait();

  // output header
  int num;
  int doublecnt = doubles-padding;
  num = fwrite(&blocks, 1, 1, stdout);
  assert(1 == num);
  num = fwrite(&warpsperblock, 1, 1, stdout);
  assert(1 == num);
  num = fwrite(&dimensionality, 1, 1, stdout);
  assert(1 == num);
  num = fwrite(&doublecnt, 4, 1, stdout);
  assert(1 == num);
  // output offset table
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int start = 0;
    if(i > 0) start = cut[i-1];
    off[i] -= ((start+1)/2*17);
    num = fwrite(&off[i], 4, 1, stdout); // chunk's compressed size in bytes
    assert(1 == num);
  }
  // output compressed data by chunk
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int offset, start = 0;
    if(i > 0) start = cut[i-1];
    offset = ((start+1)/2*17);
    // transfer compressed data back to CPU by chunk

    q.submit([&] (handler &cgh) {
      auto acc = dbufd.get_access<sycl_read>(cgh, off[i], offset);
      cgh.copy(acc, dbuf+offset);
    }).wait();

    num = fwrite(&dbuf[offset], 1, off[i], stdout);
    assert(off[i] == num);
  }

  free(cbuf);
  free(dbuf);
  free(cut);
  free(off);
}

/************************************************************************************/

static void Decompress(queue &q, int blocks, int warpsperblock, int dimensionality,
                       int doubles)
{
#ifdef DEBUG
  printf("[Decompress] allocate CPU buffers\n");
#endif
  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data, divided by chunk
  if (dbuf == NULL) { 
    fprintf(stderr, "cannot allocate dbuf\n");
  }
  ull *fbuf = (ull *)malloc(sizeof(ull) * MAX); // decompressed data
  if (fbuf == NULL) { 
    fprintf(stderr, "cannot allocate fbuf\n");
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) { 
    fprintf(stderr, "cannot allocate cut\n");
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if(off == NULL) {
    fprintf(stderr, "cannot allocate off\n");
  }

#ifdef DEBUG
  printf("[Decompress] read in offset table\n");
#endif
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num = fread(&off[i], 4, 1, stdin);
    assert(1 == num);
  }

#ifdef DEBUG
  printf("[Decompress] calculate required padding for last chunk\n");
#endif
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

#ifdef DEBUG
  printf("[Decompress] determine chunk assignments per warp\n");
#endif
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock); 
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = std::min(curr, doubles);
  }

#ifdef DEBUG
  printf("[Decompress] allocate GPU buffers\n");
#endif
  buffer<char, 1> dbufd ((doubles+1)/2*17); // compressed data
  buffer<ull, 1> fbufd (doubles); // uncompressed data
  buffer<int, 1> cutd (cut, blocks * warpsperblock); // chunk boundaries

#ifdef DEBUG
  printf("[Decompress] read in input data and divide into chunks\n");
#endif
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num, chbeg, start = 0;
    if (i > 0) start = cut[i-1];
    chbeg = ((start+1)/2*17);
    // read in this chunk of data (based on offsets)
    num = fread(&dbuf[chbeg], 1, off[i], stdin);
    assert(off[i] == num);

    // transfer the chunk to the GPU
    q.submit([&] (handler &cgh) {
      auto acc = dbufd.get_access<sycl_write>(cgh, off[i], chbeg);
      cgh.copy(dbuf + chbeg, acc);
    });
  }

#ifdef DEBUG
  printf("[Decompress] run the kernel for 100 iterations\n");
#endif

  range<1> gws (blocks * WARPSIZE * warpsperblock);
  range<1> lws (WARPSIZE * warpsperblock);

  for (int i = 0; i < 100; i++)
    q.submit([&](handler &cgh) {
      auto dbuf = dbufd.get_access<sycl_read>(cgh);
      auto fbuf = fbufd.get_access<sycl_read_write>(cgh);
      auto cut = cutd.get_access<sycl_read>(cgh);
      accessor<int, 1, sycl_read_write, access::target::local> ibufs (2 * (3 * WARPSIZE / 2), cgh);
      cgh.parallel_for(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
        DecompressionKernel(item, dimensionality,
                            dbuf.get_pointer(), fbuf.get_pointer(), cut.get_pointer(),
                            ibufs.get_pointer());
      });
    });

  // transfer result back to CPU
  q.submit([&](handler &cgh) {
    auto acc = fbufd.get_access<sycl_read>(cgh);
    cgh.copy(acc, fbuf);
  }).wait();

#ifdef DEBUG
  printf("[Decompress] output decompressed data\n");
#endif
  int num = fwrite(fbuf, 8, doubles-padding, stdout);
  assert(num == doubles-padding);

  free(dbuf);
  free(fbuf);
  free(cut);
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

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  if((3 == argc) || (4 == argc)) { /* compress */
    char dummy;
    blocks = atoi(argv[1]);
    assert((0 < blocks) && (blocks < 256));
    warpsperblock = atoi(argv[2]);
    assert((0 < warpsperblock) && (warpsperblock < 256));
    if(3 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[3]);
    }
    assert((0 < dimensionality) && (dimensionality <= WARPSIZE));

    Compress(q, blocks, warpsperblock, dimensionality);
    assert(0 == fread(&dummy, 1, 1, stdin));
  }
  else if(1 == argc) { /* decompress */
    int num, doubles;
    num = fread(&blocks, 1, 1, stdin);
    assert(1 == num);
    blocks &= 255;
    num = fread(&warpsperblock, 1, 1, stdin);
    assert(1 == num);
    warpsperblock &= 255;
    num = fread(&dimensionality, 1, 1, stdin);
    assert(1 == num);
    dimensionality &= 255;
    num = fread(&doubles, 4, 1, stdin);
    assert(1 == num);

#ifdef DEBUG
    printf("blocks=%d warps/block=%d dim=%d doubles=%d\n", blocks, warpsperblock, dimensionality, doubles);
#endif
    Decompress(q, blocks, warpsperblock, dimensionality, doubles);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s blocks warps/block (dimensionality) < file.in > file.gfc\n", argv[0]);
    fprintf(stderr, "decompress: %s < file.gfc > file.out\n", argv[0]);
  }

  return 0;
}
