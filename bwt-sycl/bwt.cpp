#include <iostream>
#include <list>
#include <sycl/sycl.hpp>
#include "bwt.hpp"

const int blockSize = 256;

void generate_table(sycl::nd_item<1> &item, int* table, int table_size, int n) {
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

void bitonic_sort_step(sycl::nd_item<1> &item, int*__restrict table, int table_size, 
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

void reconstruct_sequence(sycl::nd_item<1> &item,
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
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int table_size_bytes = table_size * sizeof(int);
  const int seq_size_bytes = n * sizeof(char);

  int *d_table = sycl::malloc_device<int>(table_size, q);
  int* table = (int*) malloc(table_size_bytes);

  int numBlocks = (table_size + blockSize - 1) / blockSize;
  sycl::range<1> gws (numBlocks * blockSize);
  sycl::range<1> lws (blockSize);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class gen>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      generate_table(item, d_table, table_size, n);
    });
  });

  char *d_sequence = sycl::malloc_device<char>(n, q);
  q.memcpy(d_sequence, sequence.c_str(), seq_size_bytes);

  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class bsort>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          bitonic_sort_step(item, d_table, table_size, j, k, d_sequence, n);
        });
      });
    }
  }

  char *d_transformed_sequence = sycl::malloc_device<char>(n, q);
  numBlocks = (n + blockSize - 1) / blockSize;
  sycl::range<1> gws2 (numBlocks * blockSize);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class restruct>(
      sycl::nd_range<1>(gws2, lws), [=] (sycl::nd_item<1> item) {
      reconstruct_sequence(item, d_table, d_sequence, d_transformed_sequence, n);
    });
  });

  char* transformed_sequence_cstr = (char*) malloc(seq_size_bytes);

  q.memcpy(transformed_sequence_cstr, d_transformed_sequence, seq_size_bytes).wait(); 

  std::string transformed_sequence(transformed_sequence_cstr, n);

  q.memcpy(table, d_table, table_size_bytes).wait(); 
  sycl::free(d_table, q);
  sycl::free(d_sequence, q);

  free(transformed_sequence_cstr);

  return std::make_pair(transformed_sequence, table);
}

std::string bwt(const std::string sequence) {
  auto data = bwt_with_suffix_array(sequence);
  free(data.second);
  return data.first;
}
