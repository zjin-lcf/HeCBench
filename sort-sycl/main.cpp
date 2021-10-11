#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "common.h"

typedef unsigned int T;

void verifySort(const T *keys, const size_t size)
{
  bool passed = true;
  for (size_t i = 0; i < size - 1; i++)
  {
    if (keys[i] > keys[i + 1])
    {
      passed = false;
#ifdef VERBOSE_OUTPUT
      std::cout << "Idx: " << i;
      std::cout << " Key: " << keys[i] << "\n";
#endif
      break;
    }
  }
  if (passed)
    std::cout << "PASS" << std::endl;
  else
    std::cout << "FAIL" << std::endl;
}

int main(int argc, char** argv) 
{

  if (argc != 3) 
  {
    printf("Usage: %s <problem size> <number of passes>\n.", argv[0]);
    return -1;
  }

  int select = atoi(argv[1]);
  int passes = atoi(argv[2]);

  // Problem Sizes
  int probSizes[4] = { 1, 8, 32, 64 };
  size_t size = probSizes[select];

  // Convert to MiB
  size = (size * 1024 * 1024) / sizeof(T);

  // Create input data on CPU
  unsigned int bytes = size * sizeof(T);

  T* h_idata = (T*) malloc (bytes); 
  T* h_odata = (T*) malloc (bytes); 

  // Initialize host memory
  std::cout << "Initializing host memory." << std::endl;
  for (int i = 0; i < size; i++)
  {
    h_idata[i] = i % 16; // Fill with some pattern
    h_odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  auto start = std::chrono::steady_clock::now();

  { // sycl scope

#ifdef USE_GPU
    gpu_selector dev_sel;
#else
    cpu_selector dev_sel;
#endif
    queue q(dev_sel);

    // Number of local work items per group
    const size_t local_wsize  = 256;
    // Number of global work items
    const size_t global_wsize = 16384; 
    // 64 work groups
    const size_t num_work_groups = global_wsize / local_wsize;

    // The radix width in bits
    const int radix_width = 4; // Changing this requires major kernel updates
    //const int num_digits = (int)pow((double)2, radix_width); // n possible digits
    const int num_digits = 16;

    const property_list props = property::buffer::use_host_ptr();
    buffer<T, 1> d_idata (h_idata, size, props);
    d_idata.set_final_data( nullptr );
    buffer<T, 1> d_odata (size); 
    d_odata.set_final_data( h_odata );
    buffer<T, 1> d_isums (num_work_groups * num_digits);

    for (int k = 0; k < passes; k++)
    {
      // Assuming an 8 bit byte.
      // shift is uint because Computecpp compiler has no operator>>(unsigned int, int);
      for (unsigned int shift = 0; shift < sizeof(T)*8; shift += radix_width)
      {
        // Like scan, we use a reduce-then-scan approach

        // But before proceeding, update the shift appropriately
        // for each kernel. This is how many bits to shift to the
        // right used in binning.

        // Also, the sort is not in place, so swap the input and output
        // buffers on each pass.
        bool even = ((shift / radix_width) % 2 == 0) ? true : false;

        q.submit([&](handler& cgh) {
            auto in = even ? d_idata.get_access<sycl_read>(cgh) : d_odata.get_access<sycl_read>(cgh);
            auto isums = d_isums.get_access<sycl_write>(cgh);
            accessor <T, 1, sycl_read_write, access::target::local> lmem (local_wsize, cgh);
            cgh.parallel_for<class reduce>(nd_range<1>(range<1>(global_wsize), range<1>(local_wsize)), [=] (nd_item<1> item) {
#include "sort_reduce.sycl"
                });
            });

        q.submit([&](handler& cgh) {
            auto isums = d_isums.get_access<sycl_read_write>(cgh);
            accessor <T, 1, sycl_read_write, access::target::local> lmem (local_wsize*2, cgh);
            accessor <T, 1, sycl_read_write, access::target::local> s_seed (1, cgh);
            cgh.parallel_for<class top_scan>(nd_range<1>(range<1>(local_wsize), range<1>(local_wsize)), [=] (nd_item<1> item) {
#include "sort_top_scan.sycl"
                });
            });

        q.submit([&](handler& cgh) {
            auto in = even ? d_idata.get_access<sycl_read_write>(cgh) : d_odata.get_access<sycl_read_write>(cgh);
            auto isums = d_isums.get_access<sycl_read_write>(cgh);
            auto out = even ? d_odata.get_access<sycl_read_write>(cgh) : d_idata.get_access<sycl_read_write>(cgh) ;
            accessor <T, 1, sycl_read_write, access::target::local> lmem (local_wsize*2, cgh);
            accessor <T, 1, sycl_read_write, access::target::local> l_scanned_seeds (16, cgh);
            accessor <T, 1, sycl_read_write, access::target::local> l_block_counts (16, cgh);
            cgh.parallel_for<class bottom_scan>(nd_range<1>(range<1>(global_wsize), range<1>(local_wsize)), [=] (nd_item<1> item) {
#include "sort_bottom_scan.sycl"
                });
            });
      }
      q.wait();
    }  // passes

  } // sycl scope

  auto end = std::chrono::steady_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double second = t / 1.e9; // Convert to seconds
  printf("Total elapsed time %.3f (s)\n", second);

  verifySort(h_odata, size);

  free(h_idata);
  free(h_odata);
  return 0;
}
