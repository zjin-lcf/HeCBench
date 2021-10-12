#include <stdio.h>
#include <stdint.h>
#include <vector>
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
void k_hashtable_insert(
  nd_item<1> &item,
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
      uint32_t expected = kEmpty;

      auto atomic_obj_ref = ext::oneapi::atomic_ref<uint32_t, 
                            ext::oneapi::memory_order::relaxed,
                            ext::oneapi::memory_scope::device,
                            access::address_space::global_space> (hashtable[slot].key);
      atomic_obj_ref.compare_exchange_strong(expected, key);

      if (expected == kEmpty || expected == key)
      {
        hashtable[slot].value = value;
        return;
      }

      slot = (slot + 1) & (kHashTableCapacity-1);
    }
  }
}

void insert_hashtable(
  queue &q,
  buffer<KeyValue, 1> &pHashTable, 
  const KeyValue*__restrict kvs, 
  uint32_t num_kvs)
{
  // Copy the keyvalues to device 
  buffer<KeyValue, 1> device_kvs (kvs, num_kvs);

  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;

  range<1> gws (gridsize * threadblocksize);
  range<1> lws (threadblocksize);
  
  q.submit([&] (handler &cgh) {
    auto table = pHashTable.get_access<sycl_read_write>(cgh);
    auto kvs = device_kvs.get_access<sycl_read>(cgh);
    cgh.parallel_for<class insert_table>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      k_hashtable_insert(item, table.get_pointer(), kvs.get_pointer(), (uint32_t)num_kvs);
    });
  });
}

// Delete each key in kvs from the hash table, if the key exists
// A deleted key is left in the hash table, but its value is set to kEmpty
// Deleted keys are not reused; once a key is assigned a slot, it never moves
void k_hashtable_delete(
  nd_item<1> &item,
  KeyValue*__restrict hashtable,
  const KeyValue*__restrict kvs,
  unsigned int numkvs)
{
  unsigned int tid = item.get_global_id(0);
  if (tid < kHashTableCapacity)
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

void delete_hashtable(queue &q, buffer<KeyValue, 1> &pHashTable, const KeyValue* kvs, uint32_t num_kvs)
{
  // Copy the keyvalues to device
  buffer<KeyValue, 1> device_kvs (kvs, num_kvs);

  // Insert all the keys into the hash table
  const int threadblocksize = 256;
  int gridsize = ((uint32_t)num_kvs + threadblocksize - 1) / threadblocksize;
  range<1> gws (gridsize * threadblocksize);
  range<1> lws (threadblocksize);
  
  q.submit([&] (handler &cgh) {
    auto table = pHashTable.get_access<sycl_read_write>(cgh);
    auto kvs = device_kvs.get_access<sycl_read>(cgh);
    cgh.parallel_for<class delete_table>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      k_hashtable_delete(item, table.get_pointer(), kvs.get_pointer(), (uint32_t)num_kvs);
    });
  });
}

// Iterate over every item in the hashtable; return non-empty key/values
void k_iterate_hashtable(
  nd_item<1> &item,
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
        auto atomic_obj_ref = ext::oneapi::atomic_ref<uint32_t, 
            ext::oneapi::memory_order::relaxed,
            ext::oneapi::memory_scope::device,
            access::address_space::global_space> (kvs_size[0]);
        uint32_t size = atomic_obj_ref.fetch_add(1);

        kvs[size] = pHashTable[tid];
      }
    }
  }
}

std::vector<KeyValue> iterate_hashtable(queue &q, buffer<KeyValue, 1> &pHashTable)
{
  buffer<uint32_t, 1> device_num_kvs (1);
  q.submit([&] (handler &cgh) {
    auto acc = device_num_kvs.get_access<sycl_discard_write>(cgh);
    cgh.fill(acc, 0u);
  });

  buffer<KeyValue, 1> device_kvs (kNumKeyValues);

  const int threadblocksize = 256;
  int gridsize = (kHashTableCapacity + threadblocksize - 1) / threadblocksize;
  range<1> gws (gridsize * threadblocksize);
  range<1> lws (threadblocksize);

  q.submit([&] (handler &cgh) {
    auto table = pHashTable.get_access<sycl_read>(cgh);
    auto kvs = device_kvs.get_access<sycl_discard_write>(cgh);
    auto kvs_size = device_num_kvs.get_access<sycl_read_write>(cgh);
    cgh.parallel_for<class iterate_table>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      k_iterate_hashtable(item,
                          table.get_pointer(), 
                          kvs.get_pointer(), 
                          kvs_size.get_pointer());
    });
  });

  uint32_t num_kvs;
  q.submit([&] (handler &cgh) {
    auto acc = device_num_kvs.get_access<sycl_read>(cgh);
    cgh.copy(acc, &num_kvs);
  }).wait();

  std::vector<KeyValue> kvs;
  kvs.resize(num_kvs);

  q.submit([&] (handler &cgh) {
    auto acc = device_kvs.get_access<sycl_read>(cgh, range<1>(num_kvs));
    cgh.copy(acc, kvs.data());
  }).wait();

  return kvs;
}

