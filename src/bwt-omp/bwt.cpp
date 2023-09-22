#include <iostream>
#include <list>
#include "bwt.hpp"

const int blockSize = 256;

void generate_table(int* table, int table_size, int n) {
  #pragma omp target teams distribute parallel for thread_limit(blockSize)
  for(int i = 0; i < table_size; i++)
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

void bitonic_sort_step(int*__restrict table, int table_size, 
                       int j, int k, const char*__restrict genome, int n) {
  #pragma omp target teams distribute parallel for thread_limit(blockSize)
  for(int i = 0; i < table_size; i++) {
    int ixj = i ^ j;
    if (i < ixj) {
      bool f = (i & k) == 0;
      int t1 = table[i];
      int t2 = table[ixj];
      if (compare_rotations(f ? t2 : t1, f ? t1 : t2, genome, n)) {
        table[i] = t2;
        table[ixj] = t1;
      }
    }
  }
}

void reconstruct_sequence(const int*__restrict table, const char*__restrict sequence, 
                          char*__restrict transformed_sequence, int n) {
  #pragma omp target teams distribute parallel for thread_limit(blockSize)
  for(int i = 0; i < n; i ++) {
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
  int table_size = n;
  // round the table size up to a power of 2 for bitonic sort
  table_size--;
  table_size |= table_size >> 1;
  table_size |= table_size >> 2;
  table_size |= table_size >> 4;
  table_size |= table_size >> 8;
  table_size |= table_size >> 16;
  table_size++;

  int* d_table = (int*) malloc(table_size * sizeof(int));
  const char* d_sequence = sequence.c_str();
  char* d_transformed_sequence = (char*) malloc(n * sizeof(char));

  #pragma omp target data map(from: d_table[0:table_size], d_transformed_sequence[0:n]) \
                          map(to: d_sequence[0:n])
  {
    generate_table(d_table, table_size, n);

    for (int k = 2; k <= table_size; k <<= 1) {
      for (int j = k >> 1; j > 0; j = j >> 1) {
        bitonic_sort_step(d_table, table_size, j, k, d_sequence, n);
      }
    }
    reconstruct_sequence(d_table, d_sequence, d_transformed_sequence, n);
  }

  std::string transformed_sequence(d_transformed_sequence, n);

  free(d_transformed_sequence);

  return std::make_pair(transformed_sequence, d_table);
}

std::string bwt(const std::string sequence) {
  auto data = bwt_with_suffix_array(sequence);
  free(data.second);
  return data.first;
}
