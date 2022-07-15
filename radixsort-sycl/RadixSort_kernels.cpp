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
sycl::uint4 scan4(nd_item<1> item, const sycl::uint4 idata, uint* ptr)
{    

  uint idx = item.get_local_id(0);

  sycl::uint4 val4 = idata;
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

sycl::uint4 rank4(nd_item<1> &item, const sycl::uint4 preds, uint* sMem, local_ptr<uint> numtrue)
{
  int localId = item.get_local_id(0);
  int localSize = item.get_local_range(0);

  sycl::uint4 address = scan4(item, preds, sMem);

  if (localId == localSize - 1) 
  {
    numtrue[0] = address.w() + preds.w();
  }
  item.barrier(access::fence_space::local_space);

  sycl::uint4 rank;
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
    sycl::uint4 lsb;
    lsb.x() = !((key.x() >> shift) & 0x1);
    lsb.y() = !((key.y() >> shift) & 0x1);
    lsb.z() = !((key.z() >> shift) & 0x1);
    lsb.w() = !((key.w() >> shift) & 0x1);

    sycl::uint4 r;

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

  key.store(globalId, keysOut);
}

//----------------------------------------------------------------------------
// Given an array with blocks sorted according to a 4-bit radix group, each 
// block counts the number of keys that fall into each radix in the group, and 
// finds the starting offset of each radix in the block.  It then writes the radix 
// counts to the counters array, and the starting offsets to the blockOffsets array.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not. "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//                                
//----------------------------------------------------------------------------
void findRadixOffsetsK(
    nd_item<1> &item,
    global_ptr<uint> keys,
    global_ptr<uint> counters,
    global_ptr<uint> blockOffsets,
    local_ptr<uint> sRadix1,
    local_ptr<uint> sStartPointers,
    const uint startbit,
    const uint totalBlocks)
{
  uint groupId = item.get_group(0);
  uint localId = item.get_local_id(0);
  uint groupSize = item.get_local_range(0);

  // uint2 radix2;
  vec<uint, 2> radix2;
  radix2.load(item.get_global_id(0), keys);


  sRadix1[2 * localId]     = (radix2.x() >> startbit) & 0xF;
  sRadix1[2 * localId + 1] = (radix2.y() >> startbit) & 0xF;

  // Finds the position where the sRadix1 entries differ and stores start 
  // index for each radix.
  if(localId < 16) 
  {
    sStartPointers[localId] = 0; 
  }
  item.barrier(access::fence_space::local_space);

  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
  {
    sStartPointers[sRadix1[localId]] = localId;
  }
  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1]) 
  {
    sStartPointers[sRadix1[localId + groupSize]] = localId + groupSize;
  }
  item.barrier(access::fence_space::local_space);

  if(localId < 16) 
  {
    blockOffsets[groupId*16 + localId] = sStartPointers[localId];
  }
  item.barrier(access::fence_space::local_space);

  // Compute the sizes of each block.
  if((localId > 0) && (sRadix1[localId] != sRadix1[localId - 1]) ) 
  {
    sStartPointers[sRadix1[localId - 1]] = 
      localId - sStartPointers[sRadix1[localId - 1]];
  }
  if(sRadix1[localId + groupSize] != sRadix1[localId + groupSize - 1] ) 
  {
    sStartPointers[sRadix1[localId + groupSize - 1]] = 
      localId + groupSize - sStartPointers[sRadix1[localId + groupSize - 1]];
  }

  if(localId == groupSize - 1) 
  {
    sStartPointers[sRadix1[2 * groupSize - 1]] = 
      2 * groupSize - sStartPointers[sRadix1[2 * groupSize - 1]];
  }
  item.barrier(access::fence_space::local_space);

  if(localId < 16) 
  {
    counters[localId * totalBlocks + groupId] = sStartPointers[localId];
  }
}

//----------------------------------------------------------------------------
// reorderData shuffles data in the array globally after the radix offsets 
// have been found. On compute version 1.1 and earlier GPUs, this code depends 
// on RadixSort::CTA_SIZE being 16 * number of radices (i.e. 16 * 2^nbits).
// 
// On compute version 1.1 GPUs ("manualCoalesce=true") this function ensures
// that all writes are coalesced using extra work in the kernel.  On later
// GPUs coalescing rules have been relaxed, so this extra overhead hurts 
// performance.  On these GPUs we set manualCoalesce=false and directly store
// the results.
//
// Template parameters are used to generate efficient code for various special cases
// For example, we have to handle arrays that are a multiple of the block size 
// (fullBlocks) differently than arrays that are not.  "loop" is used when persistent 
// CTAs are used. 
//
// By persistent CTAs we mean that we launch only as many thread blocks as can 
// be resident in the GPU and no more, rather than launching as many threads as
// we have elements. Persistent CTAs loop over blocks of elements until all work
// is complete.  This can be faster in some cases.  In our tests it is faster
// for large sorts (and the threshold is higher on compute version 1.1 and earlier
// GPUs than it is on compute version 1.2 GPUs.
//----------------------------------------------------------------------------
void reorderDataKeysOnlyK(
    nd_item<1>        &item,
    global_ptr<uint>  outKeys, 
    global_ptr<uint>  keys, 
    global_ptr<uint>  blockOffsets, 
    global_ptr<uint>  offsets, 
    local_ptr<uint> sKeys1, 
    local_ptr<uint> sOffsets, 
    local_ptr<uint> sBlockOffsets,
    const uint startbit,
    const uint numElements,
    const uint totalBlocks)
{
  uint groupId = item.get_group(0);
  uint globalId = item.get_global_id(0);
  uint localId = item.get_local_id(0);
  uint groupSize = item.get_local_range(0);

  sKeys1[2*localId] = keys[2*globalId];
  sKeys1[2*localId+1] = keys[2*globalId+1];

  if(localId < 16)  
  {
    sOffsets[localId]      = offsets[localId * totalBlocks + groupId];
    sBlockOffsets[localId] = blockOffsets[groupId * 16 + localId];
  }
  item.barrier(access::fence_space::local_space);

  uint radix = (sKeys1[localId] >> startbit) & 0xF;
  uint globalOffset = sOffsets[radix] + localId - sBlockOffsets[radix];

  if (globalOffset < numElements)
  {
    outKeys[globalOffset]   = sKeys1[localId];
  }

  radix = (sKeys1[localId + groupSize] >> startbit) & 0xF;
  globalOffset = sOffsets[radix] + localId + groupSize - sBlockOffsets[radix];

  if (globalOffset < numElements)
  {
    outKeys[globalOffset]   = sKeys1[localId + groupSize];
  }
}
