#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>

typedef unsigned int T;
typedef sycl::uint4 VECTYPE;

// CUDA kernels
#include "sort_reduce.h"
#include "sort_top_scan.h"
#include "sort_bottom_scan.h"

bool verifySort(const T *keys, const size_t size)
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
    }
  }
  std::cout << "Test ";
  if (passed)
    std::cout << "Passed" << std::endl;
  else
    std::cout << "---FAILED---" << std::endl;

  return passed;
}

int main(int argc, char** argv)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

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
  for (unsigned int i = 0; i < size; i++)
  {
    h_idata[i] = i % 16; // Fill with some pattern
    h_odata[i] = size - i;
  }

  std::cout << "Running benchmark with input array length " << size << std::endl;

  auto start = std::chrono::steady_clock::now();

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

  T* d_idata;
  T* d_odata;
  T* d_isums;

  dpct::dpct_malloc((void **)&d_idata, size * sizeof(T));
  dpct::async_dpct_memcpy(d_idata, h_idata, size * sizeof(T),
                          dpct::host_to_device);
  dpct::dpct_malloc((void **)&d_odata, size * sizeof(T));
  dpct::dpct_malloc((void **)&d_isums,
                    num_work_groups * num_digits * sizeof(T));

  T* d_in;
  T* d_out;

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
      d_in = even ? d_idata : d_odata;
      d_out = even ? d_odata : d_idata;

      {
        std::pair<dpct::buffer_t, size_t> d_in_buf_ct0 =
            dpct::get_buffer_and_offset(d_in);
        size_t d_in_offset_ct0 = d_in_buf_ct0.second;
        dpct::buffer_t d_isums_buf_ct1 = dpct::get_buffer(d_isums);
        q_ct1.submit([&](sycl::handler &cgh) {
          sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              lmem_acc_ct1(sycl::range<1>(256), cgh);
          auto d_in_acc_ct0 =
              d_in_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                  cgh);
          auto d_isums_acc_ct1 =
              d_isums_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_work_groups) *
                                    sycl::range<3>(1, 1, local_wsize),
                                sycl::range<3>(1, 1, local_wsize)),
              [=](sycl::nd_item<3> item_ct1) {
                const T *d_in_ct0 =
                    (const T *)(&d_in_acc_ct0[0] + d_in_offset_ct0);
                reduce(d_in_ct0, (T *)(&d_isums_acc_ct1[0]), size, shift,
                       item_ct1, lmem_acc_ct1.get_pointer());
              });
        });
      }
      {
        dpct::buffer_t d_isums_buf_ct0 = dpct::get_buffer(d_isums);
        q_ct1.submit([&](sycl::handler &cgh) {
          sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              lmem_acc_ct1(sycl::range<1>(512 /*256*2*/), cgh);
          sycl::accessor<T, 0, sycl::access::mode::read_write,
                         sycl::access::target::local>
              s_seed_acc_ct1(cgh);
          auto d_isums_acc_ct0 =
              d_isums_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, local_wsize),
                                             sycl::range<3>(1, 1, local_wsize)),
                           [=](sycl::nd_item<3> item_ct1) {
                             top_scan((T *)(&d_isums_acc_ct0[0]),
                                      num_work_groups, item_ct1,
                                      lmem_acc_ct1.get_pointer(),
                                      s_seed_acc_ct1.get_pointer());
                           });
        });
      }
      {
        std::pair<dpct::buffer_t, size_t> d_out_buf_ct0 =
            dpct::get_buffer_and_offset(d_out);
        size_t d_out_offset_ct0 = d_out_buf_ct0.second;
        std::pair<dpct::buffer_t, size_t> d_in_buf_ct1 =
            dpct::get_buffer_and_offset(d_in);
        size_t d_in_offset_ct1 = d_in_buf_ct1.second;
        dpct::buffer_t d_isums_buf_ct2 = dpct::get_buffer(d_isums);
        q_ct1.submit([&](sycl::handler &cgh) {
          sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              lmem_acc_ct1(sycl::range<1>(512 /*256*2*/), cgh);
          sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              l_scanned_seeds_acc_ct1(sycl::range<1>(16), cgh);
          sycl::accessor<T, 1, sycl::access::mode::read_write,
                         sycl::access::target::local>
              l_block_counts_acc_ct1(sycl::range<1>(16), cgh);
          auto d_out_acc_ct0 =
              d_out_buf_ct0.first.get_access<sycl::access::mode::read_write>(
                  cgh);
          auto d_in_acc_ct1 =
              d_in_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                  cgh);
          auto d_isums_acc_ct2 =
              d_isums_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(
              sycl::nd_range<3>(sycl::range<3>(1, 1, num_work_groups) *
                                    sycl::range<3>(1, 1, local_wsize),
                                sycl::range<3>(1, 1, local_wsize)),
              [=](sycl::nd_item<3> item_ct1) {
                T *d_out_ct0 = (T *)(&d_out_acc_ct0[0] + d_out_offset_ct0);
                const T *d_in_ct1 =
                    (const T *)(&d_in_acc_ct1[0] + d_in_offset_ct1);
                bottom_scan(d_out_ct0, d_in_ct1,
                            (const T *)(&d_isums_acc_ct2[0]), size, shift,
                            item_ct1, lmem_acc_ct1.get_pointer(),
                            l_scanned_seeds_acc_ct1.get_pointer(),
                            l_block_counts_acc_ct1.get_pointer());
              });
        });
      }
    }
    dev_ct1.queues_wait_and_throw();
  }  // passes

  dpct::dpct_memcpy(h_odata, d_out, size * sizeof(T), dpct::device_to_host);
  dpct::dpct_free(d_idata);
  dpct::dpct_free(d_odata);
  dpct::dpct_free(d_isums);

  auto end = std::chrono::steady_clock::now();
  auto t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double second = t / 1.e9; // Convert to seconds
  printf("Total elapsed time %.3f (s)\n", second);

  verifySort(h_odata, size);

  free(h_idata);
  free(h_odata);
  return 0;
}
