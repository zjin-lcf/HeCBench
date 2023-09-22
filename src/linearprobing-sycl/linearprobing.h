#pragma once

#include <sycl/sycl.hpp>

struct KeyValue
{
  uint32_t key;
  uint32_t value;
};

const uint32_t kHashTableCapacity = 64*1024*1024; //128 * 1024 * 1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xffffffff;

double insert_hashtable(sycl::queue &q,
                        KeyValue *pHashTable,
                        KeyValue *kvs,
                        uint32_t num_kvs);

double delete_hashtable(sycl::queue &q,
                        KeyValue *pHashTable,
                        KeyValue *kvs,
                        uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(sycl::queue &q, KeyValue *pHashTable);
