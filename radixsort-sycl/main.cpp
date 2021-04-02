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
#include "Scan.h"

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);
bool verifySortUint(unsigned int *keysSorted, 
    unsigned int *valuesSorted, 
    unsigned int *keysUnsorted, 
    unsigned int len);

int main(int argc, const char **argv)
{
  const unsigned int numElements = 128*128*128*2; //1048576; 
  const int keybits = 32; // bit size of uint 
  const int batchSize = 1; // only support a batch size of 1

  // Check power-of-two factorization before the scan operations start
  unsigned int arrayLength = numElements/2/CTA_SIZE*16;
  unsigned int log2L;
  unsigned int factorizationRemainder = factorRadix2(log2L, arrayLength);
  assert(factorizationRemainder == 1);

  //Check supported size range
  assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));
  assert(arrayLength > MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);

  //Check total batch size limit
  assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

  // Alloc and init some data on the host, then alloc and init GPU buffer  
  unsigned int* h_keys       = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  unsigned int* h_keysSorted = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  makeRandomUintVector(h_keys, numElements, keybits);

  unsigned int numBlocks = ((numElements % (CTA_SIZE * 4)) == 0) ? 
    (numElements / (CTA_SIZE * 4)) : (numElements / (CTA_SIZE * 4) + 1);

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<unsigned int, 1> d_keys (h_keys, numElements); 
  buffer<unsigned int, 1> d_tempKeys (numElements); 
  buffer<unsigned int, 1> d_counters (WARP_SIZE*numBlocks); 
  buffer<unsigned int, 1> d_countersSum (WARP_SIZE*numBlocks); 
  buffer<unsigned int, 1> d_blockOffsets (WARP_SIZE*numBlocks); 
  buffer<unsigned int, 1> d_buffer (arrayLength/MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE);
  d_keys.set_final_data(nullptr);

  int numIterations = 100;
  for (int i = 0; i < numIterations; i++)
  {
    radixSortKeys(q, d_keys, d_tempKeys, d_counters, d_blockOffsets, d_countersSum, 
                  d_buffer, numElements, keybits, batchSize);
  }

  // copy sorted keys to CPU 
  q.submit([&] (handler &cgh) {
    auto acc = d_keys.get_access<sycl_read>(cgh);
    cgh.copy(acc, h_keysSorted);
  });
  q.wait();

  // Check results
  bool passed = true;
  passed &= verifySortUint(h_keysSorted, NULL, h_keys, numElements);

  free(h_keys);
  free(h_keysSorted);

  // finish
  if (passed)
    printf("PASS\n");
  else 
    printf("FAIL\n");

  return 0;
}

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits)
{
  // Fill up with some random data
  int keyshiftmask = 0;
  if (keybits > 16) keyshiftmask = (1 << (keybits - 16)) - 1;
  int keymask = 0xffff;
  if (keybits < 16) keymask = (1 << keybits) - 1;

  srand(95123);
  for(unsigned int i=0; i < numElements; ++i)   
  { 
    a[i] = ((rand() & keyshiftmask)<<16) | (rand() & keymask); 
  }
}

// assumes the values were initially indices into the array, for simplicity of 
// checking correct order of values
bool verifySortUint(unsigned int *keysSorted, 
    unsigned int *valuesSorted, 
    unsigned int *keysUnsorted, 
    unsigned int len)
{
  bool passed = true;
  for(unsigned int i=0; i<len-1; ++i)
  {
    if( (keysSorted[i])>(keysSorted[i+1]) )
    {
      printf("Unordered key[%d]: %d > key[%d]: %d\n", i, keysSorted[i], i+1, keysSorted[i+1]);
      passed = false;
      break;
    }
  }

  if (valuesSorted)
  {
    for(unsigned int i=0; i<len; ++i)
    {
      if( keysUnsorted[valuesSorted[i]] != keysSorted[i] )
      {
        printf("Incorrectly sorted value[%u] (%u): %u != %u\n", 
            i, valuesSorted[i], keysUnsorted[valuesSorted[i]], keysSorted[i]);
        passed = false;
        break;
      }
    }
  }

  return passed;
}
