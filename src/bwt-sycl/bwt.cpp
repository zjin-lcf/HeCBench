#include <iostream>
#include <list>
#include <sycl/sycl.hpp>
#include "bwt.hpp"

const int blockSize = 256;

void generate_table(sycl::queue &q,
                    sycl::range<3> &gws,
                    sycl::range<3> &lws,
                    const int slm_size,
                    int* table, int table_size, int n) {
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int index = item.get_global_id(2);
      int stride = item.get_local_range(2) * item.get_group_range(2);
      for(int i = index; i < table_size; i+=stride)
        table[i] = (i < n) ? i : -1;
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
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

void bitonic_sort_step(sycl::queue &q,
                       sycl::range<3> &gws,
                       sycl::range<3> &lws,
                       const int slm_size,
                       int*__restrict table, int table_size,
                       int j, int k, const char*__restrict genome, int n) {
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int i = item.get_global_id(2);
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
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
}

void reconstruct_sequence(sycl::queue &q,
                          sycl::range<3> &gws,
                          sycl::range<3> &lws,
                          const int slm_size,
                          const int*__restrict table,
                          const char*__restrict sequence,
                          char*__restrict transformed_sequence, int n) {
  auto cgf = [&] (sycl::handler &cgh) {
    auto kfn = [=] (sycl::nd_item<3> item) {
      int index = item.get_global_id(2);
      int stride = item.get_local_range(2) * item.get_group_range(2);
      for(int i = index; i < n; i += stride) {
        transformed_sequence[i] = sequence[(n + table[i] - 1) % n];
      }
    };
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), kfn);
  };
  q.submit(cgf);
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
  sycl::range<3> gws (1, 1, numBlocks * blockSize);
  sycl::range<3> lws (1, 1, blockSize);

  generate_table(q, gws, lws, 0, d_table, table_size, n);

  char *d_sequence = sycl::malloc_device<char>(n, q);
  q.memcpy(d_sequence, sequence.c_str(), seq_size_bytes);

  for (int k = 2; k <= table_size; k <<= 1) {
    for (int j = k >> 1; j > 0; j = j >> 1) {
      bitonic_sort_step(q, gws, lws, 0, d_table, table_size, j, k, d_sequence, n);
    }
  }

  char *d_transformed_sequence = sycl::malloc_device<char>(n, q);
  numBlocks = (n + blockSize - 1) / blockSize;
  sycl::range<3> gws2 (1, 1, numBlocks * blockSize);

  reconstruct_sequence(q, gws2, lws, 0, d_table, d_sequence, d_transformed_sequence, n);

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
