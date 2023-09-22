//  Created by gangliao on 12/22/14.
//  Copyright (c) 2014 gangliao. All rights reserved.

#include "oneapi/dpl/execution"
#include "oneapi/dpl/algorithm"
#include "oneapi/dpl/numeric"
#include <sycl/sycl.hpp>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <numeric>

#include "kernels.h"

#define MAX_ALPHA 26

// number of threads in a block
#define BLOCK_SIZE 256

void suffixArray(sycl::queue &q, int*, int*, int, int);

#ifdef DEBUG
void print_suffix(const char *cc, int i)
{
  printf("%d: ", i);
  for (unsigned j = i; j < strlen(cc); j++)
    printf("%c", cc[j]);
  printf("\n");
}
#endif

bool read_data(char *filename, char *buffer, int num) {
  FILE *fh;
  fh = fopen(filename, "r");
  if (fh == NULL) {
    printf("Failed to open file %s\n", filename);
    return true;
  }
  fread(buffer, 1, num, fh);
  buffer[num] = '\0';
  fclose(fh);
  return false;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dataset> <dataset size> <repeat>\n", argv[0]);
    printf("size of dataset to evaluate (10 - 1000000)\n");
    return 1;
  }

  char* filename = argv[1];

#ifdef DEBUG
  const int n = 100;
#else
  const int n = atoi(argv[2]);
#endif

  const int repeat = atoi(argv[3]);

  //load data buffer with a genome file
  char *data = (char *) malloc ((n + 1)*sizeof(char));
  bool fail = read_data(filename, data, n);
  if (data == NULL || fail) {
    if (data) free(data);
    return 1;
  }

  std::vector<int> h_inp(n + 3);
  std::vector<int> h_SA(n + 3, 0);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_inp = sycl::malloc_device<int>(n+3, q);
  int *d_SA = sycl::malloc_device<int>(n+3, q);

  int i;
  for (i = 0; i < n; i++) h_inp[i] = (int)data[i];
  h_inp[i] = 0; h_inp[i + 1] = 0; h_inp[i + 2] = 0; //prepare for triples

  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    q.memcpy(d_inp, h_inp.data(), sizeof(int) * (n+3));
    q.memcpy(d_SA, h_SA.data(), sizeof(int) * (n+3));

    suffixArray(q, d_inp, d_SA, n, MAX_ALPHA);

    q.memcpy(h_SA.data(), d_SA, sizeof(int) * (n+3));
  }

  q.wait();

  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> time = end - start;
  printf("Average suffix array construct time (input size = %d): %f (s)\n",
         n, time.count() / repeat);

  long sum = 0;
  for (i = 0; i < n/2; i++)
    sum += (h_SA[2*i] + h_SA[2*i+1]) / abs(h_SA[2*i] - h_SA[2*i+1]);
  if (n % 2) sum -= h_SA[n-1];
  printf("checksum = 0x%lx\n", sum);

#ifdef DEBUG
  //peek sorted suffixes from data set
  for(i = 0; i < 8; i++) {
    printf("No.%d Index.", i);
    print_suffix(data, h_SA[i]);
  }

  printf("...\n...\n");

  for(i = n-8; i < n; i++) {
    printf("No.%d Index.", i);
    print_suffix(data, h_SA[i]);
  }
#endif

  sycl::free(d_inp, q);
  sycl::free(d_SA, q);
  free(data);
  return 0;
}

void suffixArray(sycl::queue &q, int *d_s, int *d_SA, int n, int K)
{
  int n0 = (n + 2) / 3,
      n2 = n / 3,
      n02 = n0 + n2;

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  int* d_s12 = sycl::malloc_device<int>(n02 + 3, q);
  q.memset(d_s12, 0, sizeof(int) * (n02+3));

  int* h_s12 = (int*) malloc (sizeof(int) * (n02 + 3));

  int* d_SA12 = sycl::malloc_device<int>(n02 + 3, q);
  q.memset(d_SA12, 0, sizeof(int) * (n02+3));

  int* d_s0 = sycl::malloc_device<int>(n0, q);
  q.memset(d_s0, 0, sizeof(int) * n0);

  int* d_SA0 = sycl::malloc_device<int>(n0, q);
  q.memset(d_SA0, 0, sizeof(int) * n0);

  int* d_scan = sycl::malloc_device<int>(n02 + 3, q);
  q.memset(d_scan, 0, sizeof(int) * (n02+3));

  sycl::range<1> lws (BLOCK_SIZE);
  sycl::range<1> gws (BLOCK_SIZE * ((n02 - 1) / BLOCK_SIZE + 1));

  // S12 initialization:
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      Init_d_s12(d_s12, n02, item);
    });
  });

  // radix sort - using SA12 to store keys
  for (int i = 2; i >= 0; i--) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        keybits(d_SA12, d_s12, d_s, n02, i, item);
      });
    });

    auto zipped_begin = oneapi::dpl::make_zip_iterator(d_SA12, d_s12);
    std::stable_sort(policy, zipped_begin, zipped_begin + n02,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

  }
  free(h_s12);

  q.memcpy(d_SA12, d_s12, sizeof(int) * (n02+3));

  // stably sort the mod 0 suffixes from SA12 by their first character
  // find lexicographic names of triples

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      InitScan(d_s, d_SA12, d_scan, n02, item);
    });
  });

  std::exclusive_scan(policy, d_scan, d_scan + n02 + 1, d_scan, 0);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      Set_suffix_rank(d_s12, d_SA12, d_scan, n02, n0, item);
    });
  });

  int max_rank;
  q.memcpy(&max_rank, &d_scan[n02], sizeof(int)).wait();

  // if max_rank is less than the size of s12, we have a repeat. repeat dc3.
  // else generate the suffix array of s12 directly
  if (max_rank < n02) {
    suffixArray(q, d_s12, d_SA12, n02, max_rank);
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        Store_unique_ranks(d_s12, d_SA12, n02, item);
      });
    });
  }
  else {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        Compute_SA_From_UniqueRank(d_s12, d_SA12, n02, item);
      });
    });
  }

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      InitScan2(d_SA12, d_scan, n0, n02, item);
    });
  });

  std::exclusive_scan(policy, d_scan, d_scan + n02, d_scan, 0);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      Set_S0(d_s0, d_SA12, d_scan, n0, n02, item);
    });
  });

  sycl::range<1> gws2 (BLOCK_SIZE * ((n0 - 1) / BLOCK_SIZE + 1));

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws2, lws), [=](sycl::nd_item<1> item) {
      keybits(d_SA0, d_s0, d_s, n0, 0, item);
    });
  });

  auto zipped_begin = oneapi::dpl::make_zip_iterator(d_SA0, d_s0);
  std::stable_sort(policy, zipped_begin, zipped_begin + n0,
    [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

  q.memcpy(d_SA0, d_s0, sizeof(int) * n0);

  // merge sorted SA0 suffixes and sorted SA12 suffixes
  sycl::range<1> gws3 (BLOCK_SIZE * ((n0 + n02 - 1) / BLOCK_SIZE + 1));
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws3, lws), [=](sycl::nd_item<1> item) {
      merge_suffixes(d_SA0, d_SA12, d_SA, d_s, d_s12, n0, n02, n, item);
    });
  });

  sycl::free(d_s12, q);
  sycl::free(d_SA12, q);
  sycl::free(d_s0, q);
  sycl::free(d_SA0, q);
  sycl::free(d_scan, q);
}
