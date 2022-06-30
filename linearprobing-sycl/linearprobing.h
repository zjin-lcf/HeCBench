#pragma once

#include "common.h"

struct KeyValue
{
  uint32_t key;
  uint32_t value;
};

const uint32_t kHashTableCapacity = 64*1024*1024; //128 * 1024 * 1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xffffffff;

double insert_hashtable(queue &q,
                        buffer<KeyValue, 1> &pHashTable,
                        buffer<KeyValue, 1> &kvs,
                        uint32_t num_kvs);

double delete_hashtable(queue &q,
                        buffer<KeyValue, 1> &pHashTable,
                        buffer<KeyValue, 1> &kvs,
                        uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(queue &q, buffer<KeyValue, 1> &pHashTable);
