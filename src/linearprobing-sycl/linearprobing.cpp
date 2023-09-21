#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <chrono>
#include "linearprobing.h"

inline uint32_t atomicCAS(uint32_t &val, uint32_t expected, uint32_t desired)
{
  uint32_t expected_value = expected;
  auto atm = sycl::atomic_ref<uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  atm.compare_exchange_strong(expected_value, desired);
  return expected_value;
}

inline uint32_t atomicAdd(uint32_t *val, uint32_t operand)
{
  auto atm = sycl::atomic_ref<uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(*val);
  return atm.fetch_add(operand);
}

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
void k_hashtable_insert(
  sycl::nd_item<1> &item,
  KeyValue*__restrict hashtable,
  const KeyValue*__restrict kvs,
  unsigned int numkvs)
{
  unsigned int tid = item.get_global_id(0);
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t value = kvs[tid].value;
    uint32_t slot = hash(key);

    while (true)
    {
      uint32_t prev = atomicCAS(hashtable[slot].key, kEmpty, key);
      if (prev == kEmpty || prev == key)
      {
        hashtable[slot].value = value;
        return;
      }

      slot = (slot + 1) & (kHashTableCapacity-1);
    }
  }
}

double insert_hashtable(
  sycl::queue &q,
  KeyValue *pHashTable,
  KeyValue *kvs,
  uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  sycl::range<1> gws (gridsize * threadblocksize);
  sycl::range<1> lws (threadblocksize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class insert_table>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      k_hashtable_insert(item, pHashTable, kvs, (uint32_t)num_kvs);
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
void k_hashtable_delete(
  sycl::nd_item<1> &item,
  KeyValue*__restrict hashtable,
  const KeyValue*__restrict kvs,
  unsigned int numkvs)
{
  unsigned int tid = item.get_global_id(0);
  if (tid < numkvs)
  {
    uint32_t key = kvs[tid].key;
    uint32_t slot = hash(key);

    while (true)
    {
      if (hashtable[slot].key == key)
      {
        hashtable[slot].value = kEmpty;
        return;
      }
      if (hashtable[slot].key == kEmpty)
      {
        return;
      }
      slot = (slot + 1) & (kHashTableCapacity - 1);
    }
  }
}

double delete_hashtable(
  sycl::queue &q,
  KeyValue *pHashTable,
  KeyValue *kvs,
  uint32_t num_kvs)
{
  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  sycl::range<1> gws (gridsize * threadblocksize);
  sycl::range<1> lws (threadblocksize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class delete_table>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      k_hashtable_delete(item, pHashTable, kvs, (uint32_t)num_kvs);
    });
  }).wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  return time;
}

// Iterate over every item in the hashtable; return non-empty key/values
void k_iterate_hashtable(
  sycl::nd_item<1> &item,
  KeyValue*__restrict pHashTable,
  KeyValue*__restrict kvs,
  uint32_t* kvs_size)
{
  unsigned int tid = item.get_global_id(0);
  if (tid < kHashTableCapacity)
  {
    if (pHashTable[tid].key != kEmpty)
    {
      uint32_t value = pHashTable[tid].value;
      if (value != kEmpty)
      {
        uint32_t size = atomicAdd(kvs_size, 1);
        kvs[size] = pHashTable[tid];
      }
    }
  }
}

std::vector<KeyValue> iterate_hashtable(sycl::queue &q, KeyValue *pHashTable)
{
  uint32_t *device_num_kvs = sycl::malloc_device<uint32_t>(1, q);
  q.memset(device_num_kvs, 0, sizeof(uint32_t));

  KeyValue *device_kvs = sycl::malloc_device<KeyValue>(kNumKeyValues, q);

  const int threadblocksize = 256;
  int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;

  sycl::range<1> gws (gridsize * threadblocksize);
  sycl::range<1> lws (threadblocksize);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class iterate_table>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      k_iterate_hashtable(item,
                          pHashTable,
                          device_kvs,
                          device_num_kvs);
    });
  });

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Kernel execution time (iterate): %f (s)\n", time * 1e-9f);

  uint32_t num_kvs;
  q.memcpy(&num_kvs, device_num_kvs, sizeof(uint32_t));

  std::vector<KeyValue> kvs;
  kvs.resize(num_kvs);

  q.memcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs);

  q.wait();
  sycl::free(device_kvs, q);
  sycl::free(device_num_kvs, q);

  return kvs;
}
