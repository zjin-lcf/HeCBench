#include <iostream>
#include <list>
#include "common.h"
#include "bwt.hpp"

const int blockSize = 256;

void generate_table(nd_item<1> &item, int* table, int table_size, int n) {
  int index = item.get_global_id(0);
  int stride = item.get_local_range(0) * item.get_group_range(0);
  for(int i = index; i < table_size; i+=stride)
    table[i] = (i < n) ? i : -1;
}

bool compare_rotations(const int& a, const int& b, const char* genome, int n) {
  if (a < 0) return false;
  if (b < 0) return true;
  for(int i = 0; i < n; i++) {
    if (genome[(a + i) % n] != genome[(b + i) % n]) {
      return genome[(a + i) % n] < genome[(b + i) % n];
    }
  }
  return false;
}

void bitonic_sort_step(nd_item<1> &item, int*__restrict table, int table_size, 
                       int j, int k, const char*__restrict genome, int n) {
  int i = item.get_global_id(0);
  int ixj = i ^ j;
  if (i < table_size && ixj > i) {
    bool f = (i & k) == 0;
    int t1 = table[i];
    int t2 = table[ixj];
    if (compare_rotations(f ? t2 : t1, f ? t1 : t2, genome, n)) {
      table[i] = t2;
      table[ixj] = t1;
    }
  }
}

void reconstruct_sequence(nd_item<1> &item,
                          const int*__restrict table,
                          const char*__restrict sequence, 
                          char*__restrict transformed_sequence, int n) {
  int index = item.get_global_id(0);
  int stride = item.get_local_range(0) * item.get_group_range(0);
  for(int i = index; i < n; i += stride) {
    transformed_sequence[i] = sequence[(n + table[i] - 1) % n];
  }
}

/* 
   returns a std::pair object
   the first item is the burrows wheeler transform of the input sequence in a std::string,
   the second item is the suffix array of the input sequence, represented as indicies of the given suffix, as an int*

   assumes input sequence already has ETX appended to it.
 */

std::pair<std::string,int*> bwt_with_suffix_array(const std::string sequence) {

  const int n = sequence.size();
  int table_size = sequence.size();
  // round the table size up to a power of 2 for bitonic sort
  table_size--;
  table_size |= table_size >> 1;
  table_size |= table_size >> 2;
  table_size |= table_size >> 4;
  table_size |= table_size >> 8;
  table_size |= table_size >> 16;
  table_size++;

#ifdef USE_GPU
  gpu_selector dev_sel;
#else
  cpu_selector dev_sel;
#endif
  queue q(dev_sel);

  buffer<int, 1> d_table (table_size);
  int* table = (int*) malloc(table_size * sizeof(int));

  int numBlocks = (table_size + blockSize - 1) / blockSize;
  range<1> gws (numBlocks * blockSize);
  range<1> lws (blockSize);

  q.submit([&] (handler &cgh) {
    auto table = d_table.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class gen>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
      generate_table(item, table.get_pointer(), table_size, n);
    });
  });

  buffer<char, 1> d_sequence (sequence.c_str(), n);

  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      q.submit([&] (handler &cgh) {
        auto table = d_table.get_access<sycl_read_write>(cgh);
        auto seq = d_sequence.get_access<sycl_read>(cgh);
        cgh.parallel_for<class bsort>(nd_range<1>(gws, lws), [=] (nd_item<1> item) {
          bitonic_sort_step(item, table.get_pointer(), table_size, j, k, seq.get_pointer(), n);
        });
      });
    }
  }

  buffer<char, 1> d_transformed_sequence (n);
  numBlocks = (n + blockSize - 1) / blockSize;
  range<1> gws2 (numBlocks * blockSize);

  q.submit([&] (handler &cgh) {
    auto table = d_table.get_access<sycl_read>(cgh);
    auto seq = d_sequence.get_access<sycl_read>(cgh);
    auto xseq = d_transformed_sequence.get_access<sycl_discard_write>(cgh);
    cgh.parallel_for<class restruct>(nd_range<1>(gws2, lws), [=] (nd_item<1> item) {
      reconstruct_sequence(item, table.get_pointer(), seq.get_pointer(), xseq.get_pointer(), n);
    });
  });
  char* transformed_sequence_cstr = (char*) malloc(n * sizeof(char));

  q.submit([&] (handler &cgh) {
    auto acc = d_transformed_sequence.get_access<sycl_read>(cgh);
    cgh.copy(acc, transformed_sequence_cstr);
  });

  std::string transformed_sequence(transformed_sequence_cstr, n);

  q.submit([&] (handler &cgh) {
    auto acc = d_table.get_access<sycl_read>(cgh);
    cgh.copy(acc, table);
  });

  q.wait();
  free(transformed_sequence_cstr);

  return std::make_pair(transformed_sequence, table);
}

std::string bwt(const std::string sequence) {
  auto data = bwt_with_suffix_array(sequence);
  free(data.second);
  return data.first;
}
