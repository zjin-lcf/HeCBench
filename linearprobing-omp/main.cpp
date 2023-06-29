#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include "linearprobing.h"

// Create random keys/values in the range [0, kEmpty)
// kEmpty is used to indicate an empty slot
std::vector<KeyValue> generate_random_keyvalues(std::mt19937& rnd, uint32_t numkvs)
{
  std::uniform_int_distribution<uint32_t> dis(0, kEmpty - 1);

  std::vector<KeyValue> kvs;
  kvs.reserve(numkvs);

  for (uint32_t i = 0; i < numkvs; i++)
  {
    uint32_t rand0 = dis(rnd);
    uint32_t rand1 = dis(rnd);
    kvs.push_back(KeyValue{rand0, rand1});
  }

  return kvs;
}

// return numshuffledkvs random items from kvs
std::vector<KeyValue> shuffle_keyvalues(std::mt19937& rnd, std::vector<KeyValue> kvs, uint32_t numshuffledkvs)
{
  std::shuffle(kvs.begin(), kvs.end(), rnd);

  std::vector<KeyValue> shuffled_kvs;
  shuffled_kvs.resize(numshuffledkvs);

  std::copy(kvs.begin(), kvs.begin() + numshuffledkvs, shuffled_kvs.begin());

  return shuffled_kvs;
}

using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

Time start_timer()
{
  return std::chrono::high_resolution_clock::now();
}

double get_elapsed_time(Time start)
{
  Time end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> d = end - start;
  std::chrono::microseconds us = std::chrono::duration_cast<std::chrono::microseconds>(d);
  return us.count() / 1000.0f;
}

void test_unordered_map(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs)
{
  Time timer = start_timer();

  printf("Timing std::unordered_map...\n");

  {
    std::unordered_map<uint32_t, uint32_t> kvs_map;
    for (auto& kv : insert_kvs)
    {
      kvs_map[kv.key] = kv.value;
    }
    for (auto& kv : delete_kvs)
    {
      auto i = kvs_map.find(kv.key);
      if (i != kvs_map.end())
        kvs_map.erase(i);
    }
  }

  double milliseconds = get_elapsed_time(timer);
  double seconds = milliseconds / 1000.0f;
  printf("Total time for std::unordered_map: %f ms (%f million keys/second)\n",
      milliseconds, kNumKeyValues / seconds / 1000000.0f);
}

void test_correctness(std::vector<KeyValue>, std::vector<KeyValue>, std::vector<KeyValue>);

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of insert batches> <number of delete batches>\n", argv[0]);
    return 1;
  }
  const uint32_t num_insert_batches = atoi(argv[1]);
  const uint32_t num_delete_batches = atoi(argv[2]);

  // To recreate the same random numbers across runs of the program, set seed to a specific
  // number instead of a number from random_device
  std::random_device rd;
  uint32_t seed = 123;
  std::mt19937 rnd(seed);  // mersenne_twister_engine

  printf("Random number generator seed = %u\n", seed);

  printf("Initializing keyvalue pairs with random numbers...\n");

  std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues);
  std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues / 2);

  // Begin test
  printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
      (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

  Time timer = start_timer();

  // Create a hash table. For linear probing, this is just an array of KeyValues
  KeyValue* pHashTable = (KeyValue*) malloc (sizeof(KeyValue) * kHashTableCapacity);

  // Initialize hash table to empty
  static_assert(kEmpty == 0xffffffff, "memset expected kEmpty=0xffffffff");
  memset(pHashTable, kEmpty, sizeof(KeyValue) * kHashTableCapacity);

  // Create key value store for hashtable insert
  uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
  KeyValue *ins_kvs = (KeyValue*) malloc (sizeof(KeyValue) * num_inserts_per_batch);

  // Create key value store for hashtable delete
  uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
  KeyValue *del_kvs = (KeyValue*) malloc (sizeof(KeyValue) * num_deletes_per_batch);

  // Create key value store for hashtable iterate
  KeyValue* iter_kvs = (KeyValue*) malloc (sizeof(KeyValue) * kNumKeyValues);

  #pragma omp target data map(to : pHashTable[0:kHashTableCapacity]) \
                          map(alloc: ins_kvs[0:num_inserts_per_batch], \
                                     del_kvs[0:num_deletes_per_batch], \
                                     iter_kvs[0:kNumKeyValues])
  {
    double total_ktime = 0.0;

    // Insert items into the hash table
    for (uint32_t i = 0; i < num_insert_batches; i++)
    {
      memcpy(ins_kvs, insert_kvs.data() + i * num_inserts_per_batch, sizeof(KeyValue) * num_inserts_per_batch);
      #pragma omp target update to (ins_kvs[0:num_inserts_per_batch])

      total_ktime += insert_hashtable(pHashTable, ins_kvs, num_inserts_per_batch);
    }
    printf("Average kernel execution time (insert): %f (s)\n", (total_ktime * 1e-9) / num_insert_batches);

    // Delete items from the hash table
    total_ktime = 0.0;
    for (uint32_t i = 0; i < num_delete_batches; i++)
    {
      memcpy(del_kvs, delete_kvs.data() + i * num_deletes_per_batch, sizeof(KeyValue) * num_deletes_per_batch);
      #pragma omp target update to (del_kvs[0:num_deletes_per_batch])

      total_ktime += delete_hashtable(pHashTable, del_kvs, num_deletes_per_batch);
    }
    printf("Average kernel execution time (delete): %f (s)\n", (total_ktime * 1e-9) / num_delete_batches);

    // Get all the key-values from the hash table
    std::vector<KeyValue> kvs = iterate_hashtable(pHashTable, iter_kvs);

    // Summarize results
    double milliseconds = get_elapsed_time(timer);
    double seconds = milliseconds / 1000.0f;
    printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n", milliseconds,
        kNumKeyValues / seconds / 1000000.0f);

    test_unordered_map(insert_kvs, delete_kvs);

    test_correctness(insert_kvs, delete_kvs, kvs);
  }

  free(pHashTable);
  free(ins_kvs);
  free(del_kvs);
  free(iter_kvs);

  printf("Success\n");

  return 0;
}
