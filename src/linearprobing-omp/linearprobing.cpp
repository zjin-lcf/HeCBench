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

  // Reinterpret KeyValue array as uint64_t for atomic operations
  uint64_t* pHashTable64 = (uint64_t*)pHashTable;
  const uint64_t kEmptyPacked = pack_kv(kEmpty, kEmpty);

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize)
  for (unsigned int tid = 0; tid < num_kvs; tid++) {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);
    uint64_t new_packed = pack_kv(key, value);

    // Linear probing with atomic operations
    // Keep trying until we successfully insert
    while (true)
    {
      // Read current slot atomically as 64-bit value
      uint64_t current_packed;
      #pragma omp atomic read
      current_packed = pHashTable64[slot];

      uint32_t current_key = unpack_key(current_packed);

      if (current_key == kEmpty) {
        // Empty slot - atomically try to claim it
        uint64_t exchanged_packed;

        #pragma omp atomic capture
        { exchanged_packed = pHashTable64[slot]; pHashTable64[slot] = new_packed; }

        // Check if we successfully claimed an empty slot
        if (exchanged_packed == kEmptyPacked) {
          // Success! We atomically claimed the slot
          break;
        }
        // Else: someone else claimed it first, check what they wrote
        uint32_t exchanged_key = unpack_key(exchanged_packed);
        if (exchanged_key == key) {
          // Someone else inserted our key - we're done
          break;
        }
        // Else: collision, continue to next slot
      } else if (current_key == key) {
        // Key already exists - we're done (duplicate insertion)
        break;
      }

      // Try next slot
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

  uint64_t* pHashTable64 = (uint64_t*)pHashTable;

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize)
  for (unsigned int tid = 0; tid < num_kvs; tid++) {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);

    // Bounded loop instead of while(true) to avoid infinite loops on GPU
    for (uint32_t iter = 0; iter < kHashTableCapacity; iter++)
    {
      // Read slot atomically
      uint64_t current_packed;
      #pragma omp atomic read
      current_packed = pHashTable64[slot];

      uint32_t current_key = unpack_key(current_packed);

      if (current_key == key)
      {
        // Mark as deleted by setting value to kEmpty, keep key intact
        uint64_t deleted_packed = pack_kv(key, kEmpty);
        #pragma omp atomic write
        pHashTable64[slot] = deleted_packed;
        break;
      }
      if (current_key == kEmpty)
      {
        break; // Key not found
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

  uint64_t* pHashTable64 = (uint64_t*)pHashTable;

  #pragma omp target teams distribute parallel for \
   num_teams(gridsize) thread_limit(threadblocksize) map(tofrom: kvs_size)
  for (unsigned int tid = 0; tid < kHashTableCapacity; tid++)
  {
    // Read slot atomically
    uint64_t current_packed;
    #pragma omp atomic read
    current_packed = pHashTable64[tid];

    uint32_t key = unpack_key(current_packed);
    uint32_t value = unpack_value(current_packed);

    if (key != kEmpty && value != kEmpty)
    {
      uint32_t size;
      #pragma omp atomic capture
      { size = kvs_size; kvs_size++; }

      device_kvs[size].key = key;
      device_kvs[size].value = value;
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
