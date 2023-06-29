// Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
// Permission is hereby granted, free of charge, to any person obtaining a copy of this
// software and associated documentation files (the "Software"), to deal in the Software
// without restriction, including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
// whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
// 
// This agreement shall be governed in all respects by the laws of the State of California and
// by the laws of the United States of America.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "options.h"
#include "scoped_ptrs.h"
using namespace aocl_utils;

#define MANUAL_VECTOR      8 
#define NUM_THREADS_PER_WG 64
#define BLOOM_1            5 
#define BLOOM_2            0x7FFFF
#define BLOOM_SIZE         14
#define docEndingTag       0xFFFFFFFF

// Params
uint block_size = 64;
uint repeat = 100;
uint total_num_docs = 256*1024;
uint total_doc_size = 0;
uint total_doc_size_no_padding = 0;

// Host Buffers
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm1;
scoped_aligned_ptr<uint> h_docWordFrequencies_dimm2;
scoped_aligned_ptr<ulong> h_profileWeights;
scoped_aligned_ptr<ulong> h_docInfo;
scoped_aligned_ptr<uint> h_isWordInProfileHash;
scoped_aligned_ptr<uint> h_startingDocID;
scoped_aligned_ptr<uint> h_numItemsPerThread;
scoped_aligned_ptr<ulong> h_profileScore;
scoped_aligned_ptr<uint> h_docSizes;

static uint m_z = 1;
static uint m_w = 1;
static uint rand_desh()
{
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    return (m_z << 16) + m_w;
}

double sampleNormal() 
{     
   double u = ((double) rand() / (RAND_MAX)) * 2 - 1;     
   double v = ((double) rand() / (RAND_MAX)) * 2 - 1;     
   double r = u * u + v * v;     
   if (r == 0 || r > 1) return sampleNormal();     
   double c = sqrt(-2 * log(r) / r);     
   return u * c; 
} 

#define DOC_LEN_SIGMA 100
#define AVG_DOC_LEN   350

uint get_doc_length()
{
   int len = sampleNormal() * DOC_LEN_SIGMA + AVG_DOC_LEN;
   if (len < 10) { len = 10; } // Arbitray lower bound;
   return (uint) len;
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

void setupData()
{
   h_startingDocID.reset( total_num_docs );
   h_numItemsPerThread.reset( total_num_docs );
   h_profileScore.reset( total_num_docs );
   h_docInfo.reset( total_num_docs );

   h_docSizes.reset( total_num_docs );

   total_doc_size = 0; 	
   total_doc_size_no_padding = 0;

   for (uint i=0; i<total_num_docs; i++) {
      uint unpadded_size = get_doc_length();
      uint size = unpadded_size & (~(2*block_size-1));
      if (unpadded_size & ((2*block_size-1))) size += 2*block_size; // Multiple of block_size
      h_startingDocID[i] = total_doc_size/2;
      h_numItemsPerThread[i] = size / (2*block_size);

      ulong start_line = total_doc_size / (2*block_size);
      ulong end_line = start_line + size / (2*block_size) - 1;

      total_doc_size += size;
      total_doc_size_no_padding += unpadded_size;
      h_docSizes[i] = unpadded_size;
      h_profileScore[i] = -1;
      h_docInfo[i] = (start_line << 32) | end_line;
   }
   h_isWordInProfileHash.reset( (1L << BLOOM_SIZE) );
   h_docWordFrequencies_dimm1.reset( total_doc_size/2 );
   h_docWordFrequencies_dimm2.reset( total_doc_size/2 );

   printf("Creating Documents total_terms=%d (no_pad=%d)\n", total_doc_size, total_doc_size_no_padding);

   for (uint i=0; i<total_doc_size/2; i++) {
      h_docWordFrequencies_dimm1[i] = docEndingTag;
      h_docWordFrequencies_dimm2[i] = docEndingTag;
   }
   for (uint doci=0; doci < total_num_docs; doci++)
   {
      uint start = h_startingDocID[doci];
      uint size = h_docSizes[doci];

      for (uint i = 0; i < size/2; i++)
      {
         uint term = (rand_desh()%((1L << 24)-1));
         uint freq = (rand_desh()%254)+1;
         h_docWordFrequencies_dimm1[start + i] = (term << 8) | freq;

         term = (rand_desh()%((1L << 24)-1));
         freq = (rand_desh()%254)+1;
         h_docWordFrequencies_dimm2[start + i] = (term << 8) | freq;
      }
      if (size%2) {
         uint term = (rand_desh()%((1L << 24)-1));
         uint freq = (rand_desh()%254)+1;
         h_docWordFrequencies_dimm1[start + size/2] = (term << 8) | freq;
      }
   }

   h_profileWeights.reset( (1L << 24) );
   for (uint i=0; i<(1L << BLOOM_SIZE); i++) {
      h_isWordInProfileHash[i] = 0x0;
   }
   printf("Creating Profile\n");
   for (uint i=0; i<(1L << 24); i++) {
      h_profileWeights[i] = 0;
   }

   for (uint i=0; i<16384; i++) {
      uint entry = (rand_desh()%(1<<24));	

      h_profileWeights[entry] = 10;

      uint hash1 = entry >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
      h_isWordInProfileHash[ hash1 >> 5 ] |= 1 << (hash1 & 0x1f);
      uint hash2 = entry & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
      h_isWordInProfileHash[ hash2 >> 5 ] |= 1 << (hash2 & 0x1f);
   }
}

void runOnCPU()
{
   // go through each document in turn, and compute the score
   scoped_aligned_ptr<ulong> cpu_profileScore;
   cpu_profileScore.reset( total_num_docs );
   uint total = 0;
   uint falsies = 0;
   for (uint doci=0; doci < total_num_docs; doci++)
   {

      cpu_profileScore[doci] = 0.0;
      uint start = h_startingDocID[doci];
      uint size = h_docSizes[doci];

      for (uint i = 0; i < size/2 + (size%2); i++)
      {
         uint curr_entry = h_docWordFrequencies_dimm1[start + i];
         uint frequency = curr_entry & 0x00ff;
         uint word_id = curr_entry >> 8;
         uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
         bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
         uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
         bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

         if (inh1 && inh2)
         {
            total++;
            if (h_profileWeights[word_id] == 0) falsies++;
            cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
         }
      }

      for (uint i = 0; i < size/2; i++)
      {
         uint curr_entry = h_docWordFrequencies_dimm2[start + i];
         uint frequency = curr_entry & 0x00ff;
         uint word_id = curr_entry >> 8;
         uint hash1 = word_id >> BLOOM_1;  //this gives me the top 16 bits of the 24bit word id
         bool inh1 = h_isWordInProfileHash[ hash1 >> 5 ] & ( 1 << (hash1 & 0x1f));
         uint hash2 = word_id & BLOOM_2;  //this gives me the bottom 16 bits of the 24bit word id
         bool inh2 = h_isWordInProfileHash[ hash2 >> 5 ] & ( 1 << (hash2 & 0x1f));

         if (inh1 && inh2)
         {
            total++;
            if (h_profileWeights[word_id] == 0) falsies++;
            cpu_profileScore[doci] += h_profileWeights[word_id] * (ulong)frequency;
         }
      }
   }

   printf( "total_access = %d , falsies = %d, percentage = %f hit= %g\n", \
		   total, falsies, total * 1.0f / total_doc_size, (total-falsies)*1.0f/total_doc_size );
   // compare the final scores
   for (uint doci = 0; doci < total_num_docs; doci++)
   {
      if (cpu_profileScore[doci] != h_profileScore[doci]) {
         printf("FAILED\n   : doc[%d] score: CPU = %lu, Device = %lu\n", \
			 doci, cpu_profileScore[doci], h_profileScore[doci]);
         return;
      }
   }
   printf( "Verification: PASS\n" );
}


ulong mulfp( ulong weight, uint freq )
{
   uint part1 = weight & 0xFFFFF;         // lower 24-bits of weight
   uint part2 = (weight >> 24) & 0xFFFF;  // next 16-bits

   uint res1 = part1 * freq;
   uint res2 = part2 * freq;

   return (ulong)res1 + (((ulong)res2) << 24);
}

int main(int argc, char** argv)
{
  Options options(argc, argv);
  // Optional argument to specify the problem size.
  if(options.has("n")) {
     total_num_docs = options.get<uint>("n");
  }
  printf("Total number of documents: %u\n", total_num_docs);

  if(options.has("p")) {
    repeat = options.get<uint>("p");
  }
  printf("Kernel execution count: %u\n", repeat);

  srand(2);
  printf("RAND_MAX: %d\n", RAND_MAX);
  printf("Allocating and setting up data\n");
  setupData();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  sycl::range<1> gws_compute (total_doc_size / 2 / MANUAL_VECTOR);
  sycl::range<1> lws_compute ((block_size / MANUAL_VECTOR));
  size_t gws_reduce = total_num_docs;
  size_t lws_reduce = block_size;

  uint *d_docWordFrequencies_dimm1 = sycl::malloc_device<uint>(total_doc_size/2, q);
  uint *d_docWordFrequencies_dimm2 = sycl::malloc_device<uint>(total_doc_size/2, q);
  uint *d_partialSums_dimm1 = sycl::malloc_device<uint>(total_doc_size/(2*block_size), q);
  uint *d_partialSums_dimm2 = sycl::malloc_device<uint>(total_doc_size/(2*block_size), q);
  ulong *d_profileWeights_dimm1 = sycl::malloc_device<ulong>((1L << 24), q);
  ulong *d_profileWeights_dimm2 = sycl::malloc_device<ulong>((1L << 24), q);
  uint *d_isWordInProfileHash = sycl::malloc_device<uint>(1L << BLOOM_SIZE, q);
  ulong *d_docInfo = sycl::malloc_device<ulong>(total_num_docs, q);
  ulong *d_profileScore = sycl::malloc_device<ulong>(total_num_docs, q);

  q.memcpy(d_docWordFrequencies_dimm1, h_docWordFrequencies_dimm1, sizeof(uint) * total_doc_size/2);
  q.memcpy(d_docWordFrequencies_dimm2, h_docWordFrequencies_dimm2, sizeof(uint) * total_doc_size/2);
  q.memcpy(d_profileWeights_dimm1, h_profileWeights, sizeof(ulong) * (1L << 24));
  q.memcpy(d_profileWeights_dimm2, h_profileWeights, sizeof(ulong) * (1L << 24));
  q.memcpy(d_isWordInProfileHash, h_isWordInProfileHash, sizeof(uint) * (1L << BLOOM_SIZE));
  q.memcpy(d_docInfo, h_docInfo, sizeof(ulong) * total_num_docs);

  q.wait();

  const double start_time = getCurrentTimestamp();
  for (uint i=0; i<repeat; i++) {
    q.submit([&] (sycl::handler &h) {
      sycl::local_accessor<ulong, 1> partial (sycl::range<1>(NUM_THREADS_PER_WG/MANUAL_VECTOR), h);
      h.parallel_for<class compute>(
      sycl::nd_range<1>(gws_compute, lws_compute), [=] (sycl::nd_item<1> item) {
        uint curr_entry[MANUAL_VECTOR];
        uint word_id[MANUAL_VECTOR];
        uint freq[MANUAL_VECTOR];
        uint hash1[MANUAL_VECTOR];
        uint hash2[MANUAL_VECTOR];
        bool is_end[MANUAL_VECTOR];
        bool make_access[MANUAL_VECTOR];

        ulong sum = 0;
        //#pragma unroll
        for (uint i=0; i<MANUAL_VECTOR; i++) {
           curr_entry[i] = d_docWordFrequencies_dimm1[item.get_global_id(0)*MANUAL_VECTOR + i]; 
           freq[i] = curr_entry[i] & 0xff;
           word_id[i] = curr_entry[i] >> 8;
           is_end[i] = curr_entry[i] == docEndingTag;
           hash1[i] = word_id[i] >> BLOOM_1;
           hash2[i] = word_id[i] & BLOOM_2;
           make_access[i] = !is_end[i] && ((d_isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
                                       && ((d_isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
           if (make_access[i]) {
              sum += mulfp(d_profileWeights_dimm1[word_id[i]],freq[i]);
           }
        }

        //#pragma unroll
        for (uint i=0; i<MANUAL_VECTOR; i++) {
           curr_entry[i] = d_docWordFrequencies_dimm2[item.get_global_id(0)*MANUAL_VECTOR + i]; 
           freq[i] = curr_entry[i] & 0xff;
           word_id[i] = curr_entry[i] >> 8;
           is_end[i] = curr_entry[i] == docEndingTag;
           hash1[i] = word_id[i] >> BLOOM_1;
           hash2[i] = word_id[i] & BLOOM_2;
           make_access[i] = !is_end[i] && ((d_isWordInProfileHash[ hash1[i] >> 5 ] >> (hash1[i] & 0x1f)) & 0x1) 
                                       && ((d_isWordInProfileHash[ hash2[i] >> 5 ] >> (hash2[i] & 0x1f)) & 0x1); 
           if (make_access[i]) {
              sum += mulfp(d_profileWeights_dimm2[word_id[i]],freq[i]);
           }
        }

        partial[item.get_local_id(0)] = sum;
        item.barrier(sycl::access::fence_space::local_space);

        if (item.get_local_id(0) == 0) {
           sycl::vec<ulong, 8> res;
           res.load(0, partial.get_pointer());
           ulong final_result = res.s0() + res.s1() + res.s2() + res.s3() +
                                res.s4() + res.s5() + res.s6() + res.s7();
           d_partialSums_dimm1[item.get_group(0)] = (uint) (final_result >> 32); 
           d_partialSums_dimm2[item.get_group(0)] = (uint) (final_result & 0xFFFFFFFF); 
        }
      });
    });

    q.submit([&] (sycl::handler &h) {
      h.parallel_for<class reduction>(
        sycl::nd_range<1>(gws_reduce, lws_reduce), [=] (sycl::nd_item<1> item) {
        ulong info = d_docInfo[item.get_global_id(0)];
        uint start = info >> 32;
        uint end = info & 0xFFFFFFFF;

        ulong total = 0;
        #pragma unroll 2
        for (uint i=start; i<=end; i++) {
           ulong upper = d_partialSums_dimm1[i];
           ulong lower = d_partialSums_dimm2[i];
           ulong sum = (upper << 32) | lower;
           total += sum;
        }

        d_profileScore[item.get_global_id(0)] = total;
      });
    });
  }
  q.wait();

  const double end_time = getCurrentTimestamp();
  double kernelExecutionTime = (end_time - start_time)/repeat;
  printf("======================================================\n");
  printf("Kernel Time = %f ms (averaged over %d times)\n", kernelExecutionTime * 1000.0f, repeat );
  printf("Throughput = %f\n", total_doc_size_no_padding / kernelExecutionTime / 1.0e+6f );

  q.memcpy(h_profileScore, d_profileScore, sizeof(ulong) * total_num_docs).wait();

  sycl::free(d_docWordFrequencies_dimm1, q);
  sycl::free(d_docWordFrequencies_dimm2, q);
  sycl::free(d_partialSums_dimm1, q);
  sycl::free(d_partialSums_dimm2, q);
  sycl::free(d_profileWeights_dimm1, q);
  sycl::free(d_profileWeights_dimm2, q);
  sycl::free(d_isWordInProfileHash, q);
  sycl::free(d_docInfo, q);
  sycl::free(d_profileScore, q);
  printf("Done\n");

  runOnCPU();
}


