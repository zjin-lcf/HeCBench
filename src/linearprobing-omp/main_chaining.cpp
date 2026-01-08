#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include "chaining.h"

// Create random keys/values in the range [0, kEmpty)
std::vector<KeyValue> generate_random_keyvalues(std::mt19937& rnd, uint32_t numkvs)
{
  std::uniform_int_distribution<uint32_t> dis(0, kEmpty_Chain - 1);

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
      milliseconds, kNumKeyValues_Chain / seconds / 1000000.0f);
}

void test_correctness_chain(std::vector<KeyValue> insert_kvs, std::vector<KeyValue> delete_kvs, std::vector<KeyValue> kvs)
{
  printf("\nTesting correctness...\n");

  // Build a hash map of all the key-values that were inserted
  std::unordered_map<uint32_t, uint32_t> inserted_kvs;
  for (auto& kv : insert_kvs)
  {
    inserted_kvs[kv.key] = kv.value;
  }

  // Remove all the key-values that were deleted
  for (auto& kv : delete_kvs)
  {
    inserted_kvs.erase(kv.key);
  }

  printf("Inserted %zu key-values, deleted %zu key-values, expected %zu remaining\n",
      insert_kvs.size(), delete_kvs.size(), inserted_kvs.size());

  // Check that all key-values returned from the hash table are in the reference map
  uint32_t num_correct = 0;
  uint32_t num_missing = 0;
  uint32_t num_duplicates = 0;
  std::unordered_set<uint32_t> seen_keys;

  for (auto& kv : kvs)
  {
    auto it = inserted_kvs.find(kv.key);
    if (it == inserted_kvs.end())
    {
      printf("ERROR: Key %u not in reference map!\n", kv.key);
      num_missing++;
    }
    else if (it->second != kv.value)
    {
      printf("ERROR: Key %u has wrong value (expected %u, got %u)\n", kv.key, it->second, kv.value);
    }
    else
    {
      num_correct++;
    }

    // Check for duplicates
    if (seen_keys.find(kv.key) != seen_keys.end())
    {
      printf("ERROR: Duplicate key %u found!\n", kv.key);
      num_duplicates++;
    }
    seen_keys.insert(kv.key);
  }

  printf("Results: %u correct, %u missing from reference, %u duplicates\n",
      num_correct, num_missing, num_duplicates);

  // Check that we didn't miss any keys
  uint32_t num_not_found = 0;
  for (auto& ref_kv : inserted_kvs)
  {
    if (seen_keys.find(ref_kv.first) == seen_keys.end())
    {
      num_not_found++;
    }
  }

  if (num_not_found > 0)
  {
    printf("ERROR: %u keys in reference map not returned by hash table!\n", num_not_found);
  }

  double correctness = (double)num_correct / (double)inserted_kvs.size() * 100.0;
  printf("Correctness: %.2f%% (%u / %zu)\n", correctness, num_correct, inserted_kvs.size());

  if (num_duplicates == 0 && num_missing == 0 && num_not_found == 0)
  {
    printf("✓ 100%% CORRECT - No race conditions!\n");
  }
  else
  {
    printf("✗ ERRORS DETECTED\n");
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of insert batches> <number of delete batches>\n", argv[0]);
    return 1;
  }
  const uint32_t num_insert_batches = atoi(argv[1]);
  const uint32_t num_delete_batches = atoi(argv[2]);

  std::random_device rd;
  uint32_t seed = 123;
  std::mt19937 rnd(seed);

  printf("=== Separate Chaining Hash Table Test ===\n");
  fflush(stdout);
  printf("Random number generator seed = %u\n", seed);
  fflush(stdout);
  printf("Buckets: %u, Max nodes: %u, Keys to insert: %u\n",
      kNumBuckets, kMaxNodes, kNumKeyValues_Chain);
  fflush(stdout);

  printf("Initializing keyvalue pairs with random numbers...\n");

  std::vector<KeyValue> insert_kvs = generate_random_keyvalues(rnd, kNumKeyValues_Chain);
  std::vector<KeyValue> delete_kvs = shuffle_keyvalues(rnd, insert_kvs, kNumKeyValues_Chain / 2);

  printf("Testing insertion/deletion of %d/%d elements into GPU hash table...\n",
      (uint32_t)insert_kvs.size(), (uint32_t)delete_kvs.size());

  Time timer = start_timer();

  // Create chaining hash table
  printf("Creating hash table...\n");
  fflush(stdout);
  ChainHashTable* table = create_chain_hashtable();
  printf("Hash table created successfully.\n");
  fflush(stdout);

  // Create key value store for hashtable insert
  uint32_t num_inserts_per_batch = (uint32_t)insert_kvs.size() / num_insert_batches;
  KeyValue *ins_kvs = (KeyValue*) malloc (sizeof(KeyValue) * num_inserts_per_batch);

  // Create key value store for hashtable delete
  uint32_t num_deletes_per_batch = (uint32_t)delete_kvs.size() / num_delete_batches;
  KeyValue *del_kvs = (KeyValue*) malloc (sizeof(KeyValue) * num_deletes_per_batch);

  // Create key value store for hashtable iterate
  KeyValue* iter_kvs = (KeyValue*) malloc (sizeof(KeyValue) * kMaxNodes);

  // Note: We don't use an outer target data region because table->buckets is a pointer-to-pointer
  // which doesn't map well. Instead, each kernel function handles its own data mapping.

  double total_ktime = 0.0;

  // Insert items into the hash table
  printf("Starting inserts (%u batches)...\n", num_insert_batches);
  fflush(stdout);
  for (uint32_t i = 0; i < num_insert_batches; i++)
  {
    printf("Insert batch %u/%u\n", i+1, num_insert_batches);
    fflush(stdout);
    memcpy(ins_kvs, insert_kvs.data() + i * num_inserts_per_batch, sizeof(KeyValue) * num_inserts_per_batch);
    total_ktime += insert_chain_hashtable(table, ins_kvs, num_inserts_per_batch);
  }
  printf("Average kernel execution time (insert): %f (s)\n", (total_ktime * 1e-9) / num_insert_batches);

  // Delete items from the hash table
  total_ktime = 0.0;
  for (uint32_t i = 0; i < num_delete_batches; i++)
  {
    memcpy(del_kvs, delete_kvs.data() + i * num_deletes_per_batch, sizeof(KeyValue) * num_deletes_per_batch);
    total_ktime += delete_chain_hashtable(table, del_kvs, num_deletes_per_batch);
  }
  printf("Average kernel execution time (delete): %f (s)\n", (total_ktime * 1e-9) / num_delete_batches);

  // Get all the key-values from the hash table
  std::vector<KeyValue> kvs = iterate_chain_hashtable(table, iter_kvs);

  // Summarize results
  double milliseconds = get_elapsed_time(timer);
  double seconds = milliseconds / 1000.0f;
  printf("Total time (including memory copies, readback, etc): %f ms (%f million keys/second)\n", milliseconds,
      kNumKeyValues_Chain / seconds / 1000000.0f);

  test_unordered_map(insert_kvs, delete_kvs);

  test_correctness_chain(insert_kvs, delete_kvs, kvs);

  destroy_chain_hashtable(table);
  free(ins_kvs);
  free(del_kvs);
  free(iter_kvs);

  printf("Success\n");

  return 0;
}
