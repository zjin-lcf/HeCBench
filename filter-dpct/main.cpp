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
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  int *data_to_filter, *filtered_data, nres = 0;
  int *d_data_to_filter, *d_filtered_data, *d_nres;

  data_to_filter = reinterpret_cast<int *>(malloc(sizeof(int) * NUM_ELEMS));

  // Generate input data.
  srand(2);
  for (int i = 0; i < NUM_ELEMS; i++) {
    data_to_filter[i] = rand() % 20;
  }

  d_data_to_filter = sycl::malloc_device<int>(NUM_ELEMS, q_ct1);
  d_filtered_data = sycl::malloc_device<int>(NUM_ELEMS, q_ct1);
  d_nres = sycl::malloc_device<int>(1, q_ct1);

  q_ct1.memcpy(d_data_to_filter, data_to_filter, sizeof(int) * NUM_ELEMS)
      .wait();
  q_ct1.memcpy(d_nres, &nres, sizeof(int)).wait();

  sycl::range<3> dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
  sycl::range<3> dimGrid(
      (NUM_ELEMS + NUM_THREADS_PER_BLOCK - 1) / NUM_THREADS_PER_BLOCK, 1, 1);

  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::accessor<int, 0, sycl::access::mode::read_write,
                   sycl::access::target::local>
        l_n_acc_ct1(cgh);

    auto dpct_global_range = dimGrid * dimBlock;

    cgh.parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                           dpct_global_range.get(0)),
            sycl::range<3>(dimBlock.get(2), dimBlock.get(1), dimBlock.get(0))),
        [=](sycl::nd_item<3> item_ct1) {
          filter(d_filtered_data, d_nres, d_data_to_filter, NUM_ELEMS, item_ct1,
                 l_n_acc_ct1.get_pointer());
        });
  });

  q_ct1.memcpy(&nres, d_nres, sizeof(int)).wait();

  filtered_data = reinterpret_cast<int *>(malloc(sizeof(int) * nres));

  q_ct1.memcpy(filtered_data, d_filtered_data, sizeof(int) * nres).wait();

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

  sycl::free(d_data_to_filter, q_ct1);
  sycl::free(d_filtered_data, q_ct1);
  sycl::free(d_nres, q_ct1);
  free(data_to_filter);
  free(filtered_data);
  free(host_filtered_data);
}
