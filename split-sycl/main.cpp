/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include "common.h"

//----------------------------------------------------------------------------
// Scans each warp in parallel ("warp-scan"), one element per thread.
// uses 2 numElements of shared memory per thread (64 = elements per warp)
//----------------------------------------------------------------------------
#define WARP_SIZE 32

uint scanwarp(nd_item<1> &item, uint val, volatile uint* sData, const int maxlevel)
{
  // The following is the same as 2 * RadixSort::WARP_SIZE * warpId + threadInWarp = 
  // 64*(threadIdx.x >> 5) + (threadIdx.x & (RadixSort::WARP_SIZE - 1))
  int localId = item.get_local_id(0);
  int idx = 2 * localId - (localId & (WARP_SIZE - 1));
  sData[idx] = 0;
  idx += WARP_SIZE;
  sData[idx] = val;     

  if (0 <= maxlevel) { sData[idx] += sData[idx - 1]; }
  if (1 <= maxlevel) { sData[idx] += sData[idx - 2]; }
  if (2 <= maxlevel) { sData[idx] += sData[idx - 4]; }
  if (3 <= maxlevel) { sData[idx] += sData[idx - 8]; }
  if (4 <= maxlevel) { sData[idx] += sData[idx -16]; }

  return sData[idx] - val;  // convert inclusive -> exclusive
}

//----------------------------------------------------------------------------
// scan4 scans 4*RadixSort::CTA_SIZE numElements in a block (4 per thread), using 
// a warp-scan algorithm
//----------------------------------------------------------------------------
cl::sycl::uint4 scan4(nd_item<1> item, const cl::sycl::uint4 idata, uint* ptr)
{    

  uint idx = item.get_local_id(0);

  cl::sycl::uint4 val4 = idata;
  uint sum[3];
  sum[0] = val4.x();
  sum[1] = val4.y() + sum[0];
  sum[2] = val4.z() + sum[1];

  uint val = val4.w() + sum[2];

  val = scanwarp(item, val, ptr, 4);
  item.barrier(access::fence_space::local_space);

  if ((idx & (WARP_SIZE - 1)) == WARP_SIZE - 1)
  {
    ptr[idx >> 5] = val + val4.w() + sum[2];
  }
  item.barrier(access::fence_space::local_space);

  if (idx < WARP_SIZE)
    ptr[idx] = scanwarp(item, ptr[idx], ptr, 2);

  item.barrier(access::fence_space::local_space);

  val += ptr[idx >> 5];

  val4.x() = val;
  val4.y() = val + sum[0];
  val4.z() = val + sum[1];
  val4.w() = val + sum[2];

  return val4;
}

cl::sycl::uint4 rank4(nd_item<1> &item, const cl::sycl::uint4 preds, uint* sMem, local_ptr<uint> numtrue)
{
  int localId = item.get_local_id(0);
  int localSize = item.get_local_range(0);

  cl::sycl::uint4 address = scan4(item, preds, sMem);

  if (localId == localSize - 1) 
  {
    numtrue[0] = address.w() + preds.w();
  }
  item.barrier(access::fence_space::local_space);

  cl::sycl::uint4 rank;
  int idx = localId*4;
  rank.x() = (preds.x()) ? address.x() : numtrue[0] + idx - address.x();
  rank.y() = (preds.y()) ? address.y() : numtrue[0] + idx + 1 - address.y();
  rank.z() = (preds.z()) ? address.z() : numtrue[0] + idx + 2 - address.z();
  rank.w() = (preds.w()) ? address.w() : numtrue[0] + idx + 3 - address.w();

  return rank;
}

void radixSortBlocksKeysK(
    nd_item<1> &item,
    global_ptr<uint> keysIn,
    global_ptr<uint> keysOut,
    const uint nbits,
    const uint startbit,
    local_ptr<uint> sMem,
    local_ptr<uint> numtrue)
{
  int globalId = item.get_global_id(0);

  vec<uint, 4> key;
  key.load(globalId, keysIn);

  item.barrier(access::fence_space::local_space);

  // radixSortBlockKeysOnly(&key, nbits, startbit, sMem, numtrue);
  int localId = item.get_local_id(0);
  int localSize = item.get_local_range(0);

  for(uint shift = startbit; shift < (startbit + nbits); ++shift)
  {
    cl::sycl::uint4 lsb;
    lsb.x() = !((key.x() >> shift) & 0x1);
    lsb.y() = !((key.y() >> shift) & 0x1);
    lsb.z() = !((key.z() >> shift) & 0x1);
    lsb.w() = !((key.w() >> shift) & 0x1);

    cl::sycl::uint4 r;

    r = rank4(item, lsb, sMem, numtrue);

    // This arithmetic strides the ranks across 4 CTA_SIZE regions
    sMem[(r.x() & 3) * localSize + (r.x() >> 2)] = key.x();
    sMem[(r.y() & 3) * localSize + (r.y() >> 2)] = key.y();
    sMem[(r.z() & 3) * localSize + (r.z() >> 2)] = key.z();
    sMem[(r.w() & 3) * localSize + (r.w() >> 2)] = key.w();
    item.barrier(access::fence_space::local_space);

    // The above allows us to read without 4-way bank conflicts:
    key.x() = sMem[localId];
    key.y() = sMem[localId +     localSize];
    key.z() = sMem[localId + 2 * localSize];
    key.w() = sMem[localId + 3 * localSize];

    item.barrier(access::fence_space::local_space);
  }

  //keysOut[globalId] = key;
  key.store(globalId, keysOut);
}

int main() {

  srand(2);
  const int N = 512;
  unsigned int *keys = (unsigned int*) malloc (N * sizeof(unsigned int));
  unsigned int *out = (unsigned int*) malloc (N * sizeof(unsigned int));
  for (int i = 0; i < N; i++)  keys[i] = rand() % 16;

  const unsigned int startbit = 0;
  const unsigned int nbits = 4;
  const unsigned threads = 128; // 1
  const unsigned teams = N/4/threads; // 1

  {
#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned int, 1> d_keys (keys, N); 
  buffer<unsigned int, 1> d_tempKeys (out, N); 

  range<1> gws (teams * threads);
  range<1> lws (threads);

  q.submit([&] (handler &cgh) {
    auto keysIn = d_keys.get_access<sycl_read>(cgh);
    auto keysOut = d_tempKeys.get_access<sycl_discard_write>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sMem(4*threads, cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> numtrue(1, cgh);
    cgh.parallel_for<class radixSort_blocksKeys>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      radixSortBlocksKeysK(item, keysIn.get_pointer(), keysOut.get_pointer(), 
                           nbits, startbit, sMem.get_pointer(), numtrue.get_pointer());

    });
  });
  }
 
  for (int i = 0; i < N; i++)  printf("%u ", out[i]);
  printf("\n");
  free(keys);
  free(out);

  return 0;
}
