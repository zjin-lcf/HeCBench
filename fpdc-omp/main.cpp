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
#include <omp.h>

#define ull unsigned long long
#define MAX (64*1024*1024)
#define WARPSIZE 32
#define min(a,b) (a) < (b) ? (a) : (b)


/* a simple impl. of clzll */
#pragma omp declare target
inline int clzll(ull num) {
  int count = 0;
  while(!(num & 0x1000000000000000ULL)) {
    count++;
    num <<= 1;
  }
  return count;
}
#pragma omp end declare target

/*
This is the GPU compression kernel, which should be launched using the block count
and warps/block:

Inputs
------
dimensionalityd: dimensionality of trace (from cmd line)
cbufd: ptr to the uncompressed data
cutd: ptr to array of chunk boundaries

Output
------
The compressed data, in dbufd 
Compressed chunk offsets for offset table, in offd
*/

void CompressionKernel(
  const int nTeams,
  const int nThreads,
  const int dimensionalityd,
  const ull *__restrict cbufd,
  char *__restrict dbufd,
  const int *__restrict cutd,
  int *__restrict offd)
{
  #pragma omp target teams num_teams(nTeams) thread_limit(nThreads)
  {
    int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum
    #pragma omp parallel 
    {
      int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
      ull diff, prev;
      int lid = omp_get_thread_num();

      // index within this warp
      lane = lid & 31;
      // index within shared prefix sum array
      iindex = lid / WARPSIZE * (3 * WARPSIZE / 2) + lane;
      ibufs[iindex] = 0;
      iindex += WARPSIZE / 2;
      lastidx = (lid / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
      // warp id
      warp = (lid + omp_get_team_num() * nThreads) / WARPSIZE;
      // prediction index within previous subchunk
      offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

      // determine start and end of chunk to compress
      start = 0;
      if (warp > 0) start = cutd[warp-1];
      term = cutd[warp];
      off = ((start+1)/2*17);

      prev = 0;
      for (int i = start + lane; i < term; i += WARPSIZE) {
        // calculate delta between value to compress and prediction
        // and negate if negative
        diff = cbufd[i] - prev;
        code = (diff >> 60) & 8;
        if (code != 0) {
          diff = -diff;
        }

        // count leading zeros in positive delta
        bcount = 8 - (clzll(diff) >> 3);
        if (bcount == 2) bcount = 3; // encode 6 lead-zero bytes as 5

        // prefix sum to determine start positions of non-zero delta bytes
        ibufs[iindex] = bcount;
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-1];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-2];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-4];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-8];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-16];
        #pragma omp barrier

        // write out non-zero bytes of delta to compressed buffer
        beg = off + (WARPSIZE/2) + ibufs[iindex-1];
        end = beg + bcount;
        for (; beg < end; beg++) {
          dbufd[beg] = diff;
          diff >>= 8;
        }

        if (bcount >= 3) bcount--; // adjust byte count for the dropped encoding
        tmp = ibufs[lastidx];
        code |= bcount;
        ibufs[iindex] = code;
        #pragma omp barrier

        // write out half-bytes of sign and leading-zero-byte count (every other thread
        // writes its half-byte and neighbor's half-byte)
        if ((lane & 1) != 0) {
          dbufd[off + (lane >> 1)] = ibufs[iindex-1] | (code << 4);
        }
        off += tmp + (WARPSIZE/2);

        // save prediction value from this subchunk (based on provided dimensionality)
        // for use in next subchunk
        prev = cbufd[i + offset];
      }

      // save final value of off, which is total bytes of compressed output for this chunk
      if (lane == 31) offd[warp] = off;
    }
  }
}

/************************************************************************************/

/*
This is the GPU decompression kernel, which should be launched using the block count
and warps/block:

Inputs
------
dimensionalityd: dimensionality of trace
dbufd: ptr to array of compressed data
cutd: ptr to array of chunk boundaries

Output
------
The decompressed data in fbufd
*/

void DecompressionKernel(
  const int nTeams,
  const int nThreads,
  const int dimensionalityd,
  const char *__restrict dbufd,
  ull *__restrict fbufd,
  const int *__restrict cutd)
{
  #pragma omp target teams num_teams(nTeams) thread_limit(nThreads)
  {
    int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum
    #pragma omp parallel 
    {
      int offset, code, bcount, off, beg, end, lane, warp, iindex, lastidx, start, term;
      ull diff, prev;
      int lid = omp_get_thread_num();

      // index within this warp
      lane = lid & 31;
      // index within shared prefix sum array
      iindex = lid / WARPSIZE * (3 * WARPSIZE / 2) + lane;
      ibufs[iindex] = 0;
      iindex += WARPSIZE / 2;
      lastidx = (lid / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
      // warp id
      warp = (lid + omp_get_team_num() * nThreads) / WARPSIZE;
      // prediction index within previous subchunk
      offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

      // determine start and end of chunk to decompress
      start = 0;
      if (warp > 0) start = cutd[warp-1];
      term = cutd[warp];
      off = ((start+1)/2*17);

      prev = 0;
      for (int i = start + lane; i < term; i += WARPSIZE) {
        // read in half-bytes of size and leading-zero count information
        if ((lane & 1) == 0) {
          code = dbufd[off + (lane >> 1)];
          ibufs[iindex] = code;
          ibufs[iindex + 1] = code >> 4;
        }
        off += (WARPSIZE/2);
        #pragma omp barrier
        code = ibufs[iindex];

        bcount = code & 7;
        if (bcount >= 2) bcount++;

        // calculate start positions of compressed data
        ibufs[iindex] = bcount;
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-1];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-2];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-4];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-8];
        #pragma omp barrier
        ibufs[iindex] += ibufs[iindex-16];
        #pragma omp barrier

        // read in compressed data (the non-zero bytes)
        beg = off + ibufs[iindex-1];
        off += ibufs[lastidx];
        end = beg + bcount - 1;
        diff = 0;
        for (; beg <= end; end--) {
          diff <<= 8;
          diff |= dbufd[end];
        }

        // negate delta if sign bit indicates it was negated during compression
        if ((code & 8) != 0) {
          diff = -diff;
        }

        // write out the uncompressed word
        fbufd[i] = prev + diff;
        #pragma omp barrier

        // save prediction for next subchunk
        prev = fbufd[i + offset];
      }
    }
  }
}



/************************************************************************************/

static void Compress(int blocks, int warpsperblock, int dimensionality)
{
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
    cut[i] = min(curr, doubles);
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

  #pragma omp target data map (to: cbuf[0:doubles], \
                                    cut[0:blocks * warpsperblock]) \
                          map (alloc: dbuf[0:(doubles+1)/2*17],\
                                       off[0:blocks * warpsperblock])
  {
 
    for (int i = 0; i < 100; i++)
      CompressionKernel(blocks, WARPSIZE*warpsperblock,
        dimensionality, cbuf, dbuf, cut, off);

    // transfer offsets back to CPU
    #pragma omp target update from (off[0:blocks * warpsperblock])

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
      #pragma omp target update from (dbuf[offset:offset+off[i]])
      num = fwrite(&dbuf[offset], 1, off[i], stdout);
      assert(off[i] == num);
    }

    free(cbuf);
    free(dbuf);
    free(cut);
    free(off);
  }
}

/************************************************************************************/

static void Decompress(int blocks, int warpsperblock, int dimensionality, int doubles)
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
    cut[i] = min(curr, doubles);
  }

#ifdef DEBUG
  printf("[Decompress] allocate GPU buffers\n");
#endif

  #pragma omp target data map (to: cut[0:blocks * warpsperblock]) \
                          map (alloc: dbuf[0:(doubles+1)/2*17]) \
                          map (from: fbuf[0:doubles])
  {

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
      #pragma omp target update to (dbuf[chbeg: chbeg+off[i]])
    }

#ifdef DEBUG
    printf("[Decompress] run the kernel for 100 iterations\n");
#endif
    for (int i = 0; i < 100; i++)
      DecompressionKernel(blocks, WARPSIZE*warpsperblock,
        dimensionality, dbuf, fbuf, cut);
  }

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

    Compress(blocks, warpsperblock, dimensionality);
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
    Decompress(blocks, warpsperblock, dimensionality, doubles);
  }
  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s blocks warps/block (dimensionality) < file.in > file.gfc\n", argv[0]);
    fprintf(stderr, "decompress: %s < file.gfc > file.out\n", argv[0]);
  }

  return 0;
}
