#pragma once

#include <omp.h>
#include <cstdint>

struct KeyValue
{
    uint32_t key;
    uint32_t value;
};

// Use 64-bit packed representation for atomic operations
// Upper 32 bits = key, Lower 32 bits = value
union PackedKV {
    uint64_t packed;
    struct {
        uint32_t value;  // Lower 32 bits
        uint32_t key;    // Upper 32 bits
    };
};

// 8M hash table capacity (2^23) - reasonable for testing
const uint32_t kHashTableCapacity = 8*1024*1024;

const uint32_t kNumKeyValues = kHashTableCapacity / 2;

const uint32_t kEmpty = 0xffffffff;
const uint32_t kLocked = 0xfffffffe;  // Special value indicating slot is being written

// Pack key and value into 64-bit integer
inline uint64_t pack_kv(uint32_t key, uint32_t value) {
    return ((uint64_t)key << 32) | (uint64_t)value;
}

// Unpack key from 64-bit integer
inline uint32_t unpack_key(uint64_t packed) {
    return (uint32_t)(packed >> 32);
}

// Unpack value from 64-bit integer
inline uint32_t unpack_value(uint64_t packed) {
    return (uint32_t)(packed & 0xFFFFFFFF);
}

KeyValue* create_hashtable();

double insert_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

double delete_hashtable(KeyValue* hashtable, const KeyValue* kvs, uint32_t num_kvs);

std::vector<KeyValue> iterate_hashtable(KeyValue* hashtable, KeyValue* device_kvs);

void destroy_hashtable(KeyValue* hashtable);
