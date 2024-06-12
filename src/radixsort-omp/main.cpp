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

#include <chrono>
#include "RadixSort.h"
#include "Scan.h"

void makeRandomUintVector(unsigned int *a, unsigned int numElements, unsigned int keybits);
bool verifySortUint(unsigned int *keysSorted, 
    unsigned int *valuesSorted, 
    unsigned int *keysUnsorted, 
    unsigned int len);

int main(int argc, const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int numIterations = atoi(argv[1]);

  const unsigned int numElements = 128*128*128*2; // 1048576; 
  const int keybits = 32;  // bit size of uint 
  const int batchSize = 1; // only support a batch size of 1
  const unsigned int numBlocks = ((numElements % (CTA_SIZE * 4)) == 0) ? 
    (numElements / (CTA_SIZE * 4)) : (numElements / (CTA_SIZE * 4) + 1);

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
  unsigned int* h_keys         = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  unsigned int* h_tempKeys     = (unsigned int*)malloc(numElements * sizeof(unsigned int));
  unsigned int* h_counters     = (unsigned int*)malloc(WARP_SIZE*numBlocks * sizeof(unsigned int));
  unsigned int* h_countersSum  = (unsigned int*)malloc(WARP_SIZE*numBlocks * sizeof(unsigned int));
  unsigned int* h_blockOffsets = (unsigned int*)malloc(WARP_SIZE*numBlocks * sizeof(unsigned int));
  unsigned int* h_buffer       = (unsigned int*)malloc(arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE
	                                               * sizeof(unsigned int));

  makeRandomUintVector(h_keys, numElements, keybits);

  // make a copy of initial keys for verification
  memcpy(h_tempKeys, h_keys, numElements*sizeof(unsigned int));

#ifdef DEBUG
  printf("#elements: %u  #blocks: %u\n", numElements, numBlocks);
  for (int i = 0; i < numElements; i++) printf("init key %d: %x\n", i, h_keys[i]);
#endif

#pragma omp target data map(tofrom: h_keys[0:numElements]) \
                        map(alloc: h_tempKeys[0:numElements], \
                 		   h_counters[0:WARP_SIZE*numBlocks], \
                 		   h_countersSum[0:WARP_SIZE*numBlocks], \
                 		   h_blockOffsets[0:WARP_SIZE*numBlocks], \
                 		   h_buffer[0:arrayLength / MAX_WORKGROUP_INCLUSIVE_SCAN_SIZE])
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; i++)
  {
    radixSortKeys(h_keys, h_tempKeys, h_counters, h_blockOffsets, h_countersSum, 
                  h_buffer, numElements, keybits, batchSize);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of radixsort: %f (s)\n", (time * 1e-9f) / numIterations);
}


#ifdef DEBUG
  for (int i = 0; i < numElements; i++) printf("sorted key %d: %x\n", i, h_keys[i]);
#endif

  // Check results
  bool passed = true;
  passed &= verifySortUint(h_keys, NULL, h_tempKeys, numElements);

  free(h_keys);
  free(h_tempKeys);
  free(h_counters);
  free(h_countersSum);
  free(h_blockOffsets);
  free(h_buffer);

  if (passed)
    printf("PASS\n");
  else 
    printf("FAIL\n");

  return passed ? 0 : 1;
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
