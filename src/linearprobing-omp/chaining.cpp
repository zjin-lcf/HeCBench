#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include "chaining.h"

// 32 bit Murmur3 hash
uint32_t hash(uint32_t k)
{
  k ^= k >> 16;
  k *= 0x85ebca6b;
  k ^= k >> 13;
  k *= 0xc2b2ae35;
  k ^= k >> 16;
  return k;  // Return full hash, bucket calculation done in caller
}

// Create hash table with separate chaining
ChainHashTable* create_chain_hashtable()
{
    ChainHashTable* table = new ChainHashTable;
    table->num_buckets = kNumBuckets;
    table->max_nodes = kMaxNodes;

    // Allocate buckets (array of indices)
    table->buckets = new uint32_t[kNumBuckets];
    for (uint32_t i = 0; i < kNumBuckets; i++) {
        table->buckets[i] = kNullIdx;  // 0xFFFFFFFF = empty
    }

    // Allocate node pool
    table->node_pool = new ChainNode[kMaxNodes];
    for (uint32_t i = 0; i < kMaxNodes; i++) {
        table->node_pool[i].next = kNullIdx;
    }

    // Allocate next_free counter
    table->next_free = new uint32_t;
    *table->next_free = 0;

    // Allocate and initialize locks (one per bucket)
    table->bucket_locks = new omp_lock_t[kNumBuckets];
    for (uint32_t i = 0; i < kNumBuckets; i++) {
        omp_init_lock(&table->bucket_locks[i]);
    }

    return table;
}

void destroy_chain_hashtable(ChainHashTable* table)
{
    // Destroy all locks
    for (uint32_t i = 0; i < table->num_buckets; i++) {
        omp_destroy_lock(&table->bucket_locks[i]);
    }
    delete[] table->bucket_locks;

    delete[] table->buckets;
    delete[] table->node_pool;
    delete table->next_free;
    delete table;
}

// Insert using lock-based separate chaining (100% correct with locks)
double insert_chain_hashtable(ChainHashTable* table, const KeyValue* kvs, uint32_t num_kvs)
{
    const int threadblocksize = 256;
    int gridsize = (num_kvs + threadblocksize - 1) / threadblocksize;

    auto start = std::chrono::steady_clock::now();

    uint32_t* buckets = table->buckets;
    ChainNode* node_pool = table->node_pool;
    uint32_t* next_free = table->next_free;
    omp_lock_t* bucket_locks = table->bucket_locks;
    uint32_t num_buckets = table->num_buckets;
    uint32_t max_nodes = table->max_nodes;

    #pragma omp target teams distribute parallel for \
     num_teams(gridsize) thread_limit(threadblocksize) \
     map(to: kvs[0:num_kvs]) \
     map(tofrom: buckets[0:num_buckets], node_pool[0:max_nodes], \
                 next_free[0:1], bucket_locks[0:num_buckets])
    for (uint32_t tid = 0; tid < num_kvs; tid++) {
        uint32_t key = kvs[tid].key;
        uint32_t value = kvs[tid].value;
        uint32_t bucket = hash(key) % num_buckets;

        // Allocate node from pool atomically
        uint32_t node_idx;
        #pragma omp atomic capture
        { node_idx = *next_free; (*next_free)++; }

        if (node_idx >= max_nodes) {
            // Out of nodes - should not happen with proper sizing
            continue;
        }

        // Initialize the new node
        node_pool[node_idx].key = key;
        node_pool[node_idx].value = value;

        // Lock the bucket before modifying its list
        omp_set_lock(&bucket_locks[bucket]);

        // Prepend node to list (safe inside lock, no atomics needed)
        node_pool[node_idx].next = buckets[bucket];
        buckets[bucket] = node_idx;

        // Unlock the bucket
        omp_unset_lock(&bucket_locks[bucket]);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return time;
}

// Delete using separate chaining with locks
double delete_chain_hashtable(ChainHashTable* table, const KeyValue* kvs, uint32_t num_kvs)
{
    const int threadblocksize = 256;
    int gridsize = (num_kvs + threadblocksize - 1) / threadblocksize;

    auto start = std::chrono::steady_clock::now();

    uint32_t* buckets = table->buckets;
    uint32_t num_buckets = table->num_buckets;
    uint32_t max_nodes = table->max_nodes;
    ChainNode* node_pool = table->node_pool;
    omp_lock_t* bucket_locks = table->bucket_locks;

    #pragma omp target teams distribute parallel for \
     num_teams(gridsize) thread_limit(threadblocksize) \
     map(to: kvs[0:num_kvs]) \
     map(tofrom: buckets[0:num_buckets], node_pool[0:max_nodes], bucket_locks[0:num_buckets])
    for (uint32_t tid = 0; tid < num_kvs; tid++) {
        uint32_t key = kvs[tid].key;
        uint32_t bucket = hash(key) % num_buckets;

        // Lock bucket before traversing (prevents concurrent modifications)
        omp_set_lock(&bucket_locks[bucket]);

        // Traverse list to find and mark node as deleted
        uint32_t node_idx = buckets[bucket];
        while (node_idx != kNullIdx) {
            if (node_pool[node_idx].key == key) {
                // Mark as deleted by setting value to kEmpty_Chain
                node_pool[node_idx].value = kEmpty_Chain;
                break;
            }
            node_idx = node_pool[node_idx].next;
        }

        // Unlock bucket
        omp_unset_lock(&bucket_locks[bucket]);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return time;
}

// Iterate over all non-deleted entries with locks
std::vector<KeyValue> iterate_chain_hashtable(ChainHashTable* table, KeyValue* device_kvs)
{
    uint32_t kvs_size = 0;
    const int threadblocksize = 256;
    int gridsize = (table->num_buckets + threadblocksize - 1) / threadblocksize;

    auto start = std::chrono::steady_clock::now();

    uint32_t* buckets = table->buckets;
    ChainNode* node_pool = table->node_pool;
    uint32_t num_buckets = table->num_buckets;
    uint32_t max_nodes = table->max_nodes;
    omp_lock_t* bucket_locks = table->bucket_locks;

    #pragma omp target teams distribute parallel for \
     num_teams(gridsize) thread_limit(threadblocksize) \
     map(tofrom: kvs_size, buckets[0:num_buckets], node_pool[0:max_nodes], \
                 device_kvs[0:max_nodes], bucket_locks[0:num_buckets])
    for (uint32_t bid = 0; bid < num_buckets; bid++) {
        // Lock bucket before traversing
        omp_set_lock(&bucket_locks[bid]);

        // Traverse list
        uint32_t node_idx = buckets[bid];
        while (node_idx != kNullIdx) {
            uint32_t key = node_pool[node_idx].key;
            uint32_t value = node_pool[node_idx].value;

            if (value != kEmpty_Chain) {
                // Add to output (still need atomic for shared counter)
                uint32_t idx;
                #pragma omp atomic capture
                { idx = kvs_size; kvs_size++; }

                device_kvs[idx].key = key;
                device_kvs[idx].value = value;
            }

            node_idx = node_pool[node_idx].next;
        }

        // Unlock bucket
        omp_unset_lock(&bucket_locks[bid]);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Kernel execution time (iterate chaining): %f (s)\n", time * 1e-9f);

    uint32_t num_kvs = kvs_size;

    std::vector<KeyValue> kvs;
    kvs.resize(num_kvs);

    #pragma omp target update from (device_kvs[0:num_kvs])
    memcpy(kvs.data(), device_kvs, sizeof(KeyValue) * num_kvs);

    return kvs;
}
