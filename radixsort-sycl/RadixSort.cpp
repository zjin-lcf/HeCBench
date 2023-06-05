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
    sycl::queue &q,
    uint *d_keys,
    uint *d_tempKeys,
    const uint nbits,
    const uint startbit,
    const uint numElements)
{
  uint totalBlocks = numElements/4/CTA_SIZE;
  sycl::range<1> gws (CTA_SIZE*totalBlocks);
  sycl::range<1> lws (CTA_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint, 1> sMem(sycl::range<1>(4*CTA_SIZE), cgh);
    sycl::local_accessor<uint, 0> numtrue(cgh);
    cgh.parallel_for<class radixSort_blocksKeys>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      radixSortBlocksKeysK(item, d_keys, d_tempKeys,
                           nbits, startbit, sMem.get_pointer(), numtrue);
    });
  });
}

void findRadixOffsets(
  sycl::queue &q,
  uint *d_tempKeys,
  uint *d_counters,
  uint *d_blockOffsets,
  const uint startbit,
  const uint numElements)
{
  uint totalBlocks = numElements/2/CTA_SIZE;

  sycl::range<1> gws (CTA_SIZE*totalBlocks);
  sycl::range<1> lws (CTA_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint, 1> sRadix1(sycl::range<1>(2*CTA_SIZE), cgh);
    sycl::local_accessor<uint, 1> sStartPointers(sycl::range<1>(16), cgh);
    cgh.parallel_for<class find_radix_Offsets>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      findRadixOffsetsK(item, d_tempKeys, d_counters,
                        d_blockOffsets, sRadix1.get_pointer(),
                        sStartPointers.get_pointer(),
                        startbit, totalBlocks);
    });
  });
}

void reorderDataKeysOnly(
   sycl::queue  &q,
   uint *d_keys,
   uint *d_tempKeys,
   uint *d_blockOffsets,
   uint *d_countersSum,
   const uint startbit,
   const uint numElements)
{
  uint totalBlocks = numElements/2/CTA_SIZE;
  sycl::range<1> gws (CTA_SIZE*totalBlocks);
  sycl::range<1> lws (CTA_SIZE);

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<uint, 1> sKeys1(sycl::range<1>(2*CTA_SIZE), cgh);
    sycl::local_accessor<uint, 1> sOffsets(sycl::range<1>(16), cgh);
    sycl::local_accessor<uint, 1> sBlockOffsets(sycl::range<1>(16), cgh);
    cgh.parallel_for<class reorder_data_keys_only>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      reorderDataKeysOnlyK(item,
                           d_keys,
                           d_tempKeys,
                           d_blockOffsets,
                           d_countersSum,
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
void radixSortStepKeysOnly(sycl::queue &q,
                           uint *d_keys,
                           uint *d_tempKeys,
                           uint *d_counters,
                           uint *d_blockOffsets,
                           uint *d_countersSum,
                           uint *d_buffer,
                           const uint nbits,
                           const uint startbit,
                           const uint numElements,
                           const uint batchSize)
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
void radixSortKeys(sycl::queue &q,
                   uint *d_keys,
                   uint *d_tempKeys,
                   uint *d_counters,
                   uint *d_blockOffsets,
                   uint *d_countersSum,
                   uint *d_buffer,
                   const uint numElements,
                   const uint keyBits,
                   const uint batchSize)
{
  int i = 0;
  while (keyBits > i*bitStep)
  {
    radixSortStepKeysOnly(q, d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum,
                          d_buffer, bitStep, i*bitStep, numElements, batchSize);
    i++;
  }
}
