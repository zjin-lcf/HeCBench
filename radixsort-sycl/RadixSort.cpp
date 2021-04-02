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

#include "RadixSort.h"
#include "RadixSort_kernels.cpp"

//----------------------------------------------------------------------------
// Wrapper for the kernels of the four steps
//----------------------------------------------------------------------------
void radixSortBlocksKeysOnly(
    queue &q,
    buffer<unsigned int> &d_keys, 
    buffer<unsigned int> &d_tempKeys, 
    const unsigned int nbits, 
    const unsigned int startbit, 
    const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/4/CTA_SIZE;
  range<1> gws (CTA_SIZE*totalBlocks);
  range<1> lws (CTA_SIZE);

  q.submit([&] (handler &cgh) {
    auto keysIn = d_keys.get_access<sycl_read>(cgh);
    auto keysOut = d_tempKeys.get_access<sycl_discard_write>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sMem(4*CTA_SIZE, cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> numtrue(1, cgh);
    cgh.parallel_for<class radixSort_blocksKeys>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      radixSortBlocksKeysK(item, keysIn.get_pointer(), keysOut.get_pointer(), 
                           nbits, startbit, sMem.get_pointer(), numtrue.get_pointer());

    });
  });
}

void findRadixOffsets(
  queue &q,
  buffer<unsigned int> d_tempKeys, 
  buffer<unsigned int> d_counters, 
  buffer<unsigned int> d_blockOffsets, 
  const unsigned int startbit, 
  const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/2/CTA_SIZE;

  range<1> gws (CTA_SIZE*totalBlocks);
  range<1> lws (CTA_SIZE);

  q.submit([&] (handler &cgh) {
    auto keys = d_tempKeys.get_access<sycl_read>(cgh);
    auto counters = d_counters.get_access<sycl_discard_write>(cgh);
    auto blockOffsets = d_blockOffsets.get_access<sycl_discard_write>(cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sRadix1(2*CTA_SIZE, cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sStartPointers(16, cgh);
    cgh.parallel_for<class find_radix_Offsets>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      findRadixOffsetsK(item, keys.get_pointer(), counters.get_pointer(),
                        blockOffsets.get_pointer(), sRadix1.get_pointer(), 
                        sStartPointers.get_pointer(),
                        startbit, totalBlocks);
    });
  });
}

void reorderDataKeysOnly(
   queue        &q,
   buffer<uint> d_keys, 
   buffer<uint> d_tempKeys, 
   buffer<uint> d_blockOffsets, 
   buffer<uint> d_countersSum, 
   const unsigned int startbit, 
   const unsigned int numElements)
{
  unsigned int totalBlocks = numElements/2/CTA_SIZE;
  range<1> gws (CTA_SIZE*totalBlocks);
  range<1> lws (CTA_SIZE);

  q.submit([&] (handler &cgh) {
    auto keys = d_keys.get_access<sycl_write>(cgh);
    auto tempKeys = d_tempKeys.get_access<sycl_read>(cgh);
    auto blockOffsets = d_blockOffsets.get_access<sycl_read>(cgh);
    auto countersSum = d_countersSum.get_access<sycl_read>(cgh);

    accessor<unsigned int, 1, sycl_read_write, access::target::local> sKeys1(2*CTA_SIZE, cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sOffsets(16, cgh);
    accessor<unsigned int, 1, sycl_read_write, access::target::local> sBlockOffsets(16, cgh);
    cgh.parallel_for<class reorder_data_keys_only>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      reorderDataKeysOnlyK(item, keys.get_pointer(),
                        tempKeys.get_pointer(),
                        blockOffsets.get_pointer(),
                        countersSum.get_pointer(),
                        sKeys1.get_pointer(),
                        sOffsets.get_pointer(),
                        sBlockOffsets.get_pointer(),
                        startbit,
                        numElements,
                        totalBlocks);
    });
  });
}

//----------------------------------------------------------------------------
// Perform one step of the radix sort.  Sorts by "nbits" key bits per step, 
// starting at startbit.
//----------------------------------------------------------------------------
void radixSortStepKeysOnly(queue &q,
                           buffer<unsigned int> &d_keys, 
                           buffer<unsigned int> &d_tempKeys, 
                           buffer<unsigned int> &d_counters, 
                           buffer<unsigned int> &d_blockOffsets, 
                           buffer<unsigned int> &d_countersSum, 
                           buffer<unsigned int> &d_buffer, 
                           const unsigned int nbits, 
                           const unsigned int startbit, 
                           const unsigned int numElements,
                           const unsigned int batchSize)
{
  // Four step algorithms from Satish, Harris & Garland
  radixSortBlocksKeysOnly(q, d_keys, d_tempKeys, nbits, startbit, numElements);

  findRadixOffsets(q, d_tempKeys, d_counters, d_blockOffsets, startbit, numElements);

  scanExclusiveLarge(q, d_countersSum, d_counters, d_buffer, batchSize, numElements/2/CTA_SIZE*16, numElements);

  reorderDataKeysOnly(q, d_keys, d_tempKeys, d_blockOffsets, d_countersSum, startbit, numElements);
}


//----------------------------------------------------------------------------
// Main key-only radix sort function.  Sorts in place in the keys and values 
// arrays, but uses the other device arrays as temporary storage.  All pointer 
// parameters are device pointers.  Uses cudppScan() for the prefix sum of
// radix counters.
//----------------------------------------------------------------------------
void radixSortKeys(queue &q,
                   buffer<unsigned int> &d_keys, 
                   buffer<unsigned int> &d_tempKeys, 
                   buffer<unsigned int> &d_counters, 
                   buffer<unsigned int> &d_blockOffsets, 
                   buffer<unsigned int> &d_countersSum, 
                   buffer<unsigned int> &d_buffer, 
                   const unsigned int numElements, 
                   const unsigned int keyBits,
                   const unsigned int batchSize)
{
  int i = 0;
  while (keyBits > i*bitStep) 
  {
    radixSortStepKeysOnly(q, d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
                          d_buffer, bitStep, i*bitStep, numElements, batchSize);
    i++;
  }
}
