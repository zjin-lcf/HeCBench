#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include "linearprobing.h"

// 32 bit Murmur3 hash
uint32_t hash(uint32_t k)
{
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k & (kHashTableCapacity-1);
}

// Insert the key/values in kvs into the hashtable
double insert_hashtable(KeyValue*__restrict pHashTable,
                        const KeyValue*__restrict kvs,
                        uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  auto start = std::chrono::steady_clock::now();

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize)
  for (unsigned int tid = 0; tid < num_kvs; tid++) {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);

    while (true)
    {
      uint32_t prev;

      #pragma omp atomic capture
      //#pragma omp critical
      {
        prev = pHashTable[slot].key;
        pHashTable[slot].key = (prev == kEmpty) ? key : prev;
      }
      if (prev == kEmpty || prev == key)
      {
        pHashTable[slot].value = value;
        break; //return;
      }

      slot = (slot + 1) & (kHashTableCapacity-1);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  return time;
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
double delete_hashtable(KeyValue* pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  auto start = std::chrono::steady_clock::now();

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize)
  for (unsigned int tid = 0; tid < num_kvs; tid++) {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);

    while (true)
    {
      if (pHashTable[slot].key == key)
      {
        pHashTable[slot].value = kEmpty;
        break; //return;
      }
      if (pHashTable[slot].key == kEmpty)
      {
        break; //return;
      }
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

// Iterate over every item in the hashtable; return non-empty key/values
std::vector<KeyValue> iterate_hashtable(KeyValue* pHashTable, KeyValue *device_kvs)
{
  uint32_t kvs_size = 0;
  const int threadblocksize = 256;
  int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;

  auto start = std::chrono::steady_clock::now();

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize) map(tofrom: kvs_size)
  for (unsigned int tid = 0; tid < kHashTableCapacity; tid++) 
  {
    if (pHashTable[tid].key != kEmpty) 
    {
      uint32_t value = pHashTable[tid].value;
      if (value != kEmpty)
      {
        uint32_t size;
        #pragma omp atomic capture
        { size = kvs_size; kvs_size++; }
        device_kvs[size] = pHashTable[tid];
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (iterate): %f (s)\n", time * 1e-9f);

  uint32_t num_kvs = kvs_size;

  std::vector<KeyValue> kvs;
  kvs.resize(num_kvs);

  #pragma omp target update from (device_kvs[0:num_kvs])
  memcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs);

  return kvs;
}
