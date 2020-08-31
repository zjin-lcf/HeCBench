/**
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#define NUM_ELEMS 100000000
#define NUM_THREADS_PER_BLOCK 256


void filter(int *dst, int *nres, const int* src, int n,
            sycl::nd_item<3> item_ct1, int *l_n) {

  int i = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
          item_ct1.get_local_id(2);

  // zero the counter
  if (item_ct1.get_local_id(2) == 0)
    *l_n = 0;
  item_ct1.barrier();

  // get the value, evaluate the predicate, and
  // increment the counter if needed
  int d, pos;

  if(i < n) {
    d = src[i];
    if(d > 0)
      pos = sycl::atomic<int, sycl::access::address_space::local_space>(
                sycl::local_ptr<int>(l_n))
                .fetch_add(1);
  }
  item_ct1.barrier();

  // leader increments the global counter
  if (item_ct1.get_local_id(2) == 0)
    *l_n = sycl::atomic<int>(sycl::global_ptr<int>(nres)).fetch_add(*l_n);
  item_ct1.barrier();

  // threads with true predicates write their elements
  if(i < n && d > 0) {
    pos += *l_n; // increment local pos by global counter
    dst[pos] = d;
  }
  item_ct1.barrier();
}


int main(int argc, char **argv) {
  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  srand(2);
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % 20;
  }

  dpct::dpct_malloc(&d_data_to_filter, sizeof(int) * NUM_ELEMS);
  dpct::dpct_malloc(&d_filtered_data, sizeof(int) * NUM_ELEMS);
  dpct::dpct_malloc(&d_nres, sizeof(int));

  dpct::dpct_memcpy(d_data_to_filter, data_to_filter, sizeof(int) * NUM_ELEMS,
                    dpct::host_to_device);
  dpct::dpct_memcpy(d_nres, &nres, sizeof(int), dpct::host_to_device);

  sycl::range<3> dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  sycl::range<3> dimGrid(
      (NUM_ELEMS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK, 1, 1);

  {
    dpct::buffer_t d_filtered_data_buf_ct0 = dpct::get_buffer(d_filtered_data);
    dpct::buffer_t d_nres_buf_ct1 = dpct::get_buffer(d_nres);
    dpct::buffer_t d_data_to_filter_buf_ct2 =
        dpct::get_buffer(d_data_to_filter);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<int, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          l_n_acc_ct1(cgh);
      auto d_filtered_data_acc_ct0 =
          d_filtered_data_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_nres_acc_ct1 =
          d_nres_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_data_to_filter_acc_ct2 =
          d_data_to_filter_buf_ct2.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = dimGrid * dimBlock;

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                           dpct_global_range.get(1),
                                           dpct_global_range.get(0)),
                            sycl::range<3>(dimBlock.get(2), dimBlock.get(1),
                                           dimBlock.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            filter((int *)(&d_filtered_data_acc_ct0[0]),
                   (int *)(&d_nres_acc_ct1[0]),
                   (const int *)(&d_data_to_filter_acc_ct2[0]), NUM_ELEMS,
                   item_ct1, l_n_acc_ct1.get_pointer());
          });
    });
  }

  dpct::dpct_memcpy(&nres, d_nres, sizeof(int), dpct::device_to_host);

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  dpct::dpct_memcpy(filtered_data, d_filtered_data, sizeof(int) * nres,
                    dpct::device_to_host);

  int *host_filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate host output with host filtering code.
  int host_flt_count = 0;
  for (int i = 0; i < NUM_ELEMS; i++) {
    if (data_to_filter[i] > 0) {
      host_filtered_data[host_flt_count++] = data_to_filter[i];
    }
  }

  printf("\nFilter using shared memory %s \n",
         host_flt_count == nres ? "PASSED" : "FAILED");

  dpct::dpct_free(d_data_to_filter);
  dpct::dpct_free(d_filtered_data);
  dpct::dpct_free(d_nres);
  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
