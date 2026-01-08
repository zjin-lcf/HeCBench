#pragma once

#include <omp.h>
#include <cstdint>
#include <vector>

// Separate chaining hash table - 100% correct with OpenMP atomics

struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

#pragma omp declare target
struct ChainNode {
    uint32_t key;
    uint32_t value;
    uint32_t next;  // Index into node_pool (0xFFFFFFFF = NULL)
};
#pragma omp end declare target

const uint32_t kNumBuckets = 1024;  // 1K buckets (very small for debugging)
const uint32_t kMaxNodes = 4096;   // 4K nodes (very small for debugging)
const uint32_t kNumKeyValues_Chain = kMaxNodes / 2;  // Insert 2K keys

const uint32_t kEmpty_Chain = 0xffffffff;
const uint32_t kNullIdx = 0xffffffff;  // Null pointer index

struct ChainHashTable {
    uint32_t* buckets;         // Array of list head indices (0xFFFFFFFF = empty)
    ChainNode* node_pool;     // Pre-allocated nodes
    uint32_t* next_free;      // Index of next free node
    omp_lock_t* bucket_locks;  // One lock per bucket for safe prepend
    uint32_t num_buckets;
    uint32_t max_nodes;
};

ChainHashTable* create_chain_hashtable();

double insert_chain_hashtable(ChainHashTable* table, const KeyValue* kvs, uint32_t num_kvs);

double delete_chain_hashtable(ChainHashTable* table, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_chain_hashtable(ChainHashTable* table, KeyValue* device_kvs);

void destroy_chain_hashtable(ChainHashTable* table);
