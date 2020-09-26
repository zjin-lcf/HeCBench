#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel.cpp"
#include "./kernel_wrapper.h"        // (in current directory)

void 
kernel_wrapper(record *records,
    long records_mem,
    knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    int *keys,
    record *ans)
{

  long long offload_start = get_time();

  int numBlocks;
  numBlocks = count;                  // max # of blocks can be 65,535
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);

  //==================================================50
  //  recordsD
  //==================================================50

  record *recordsD;
  dpct::dpct_malloc((void **)&recordsD, records_mem);

  //==================================================50
  //  knodesD
  //==================================================50

  knode *knodesD;
  dpct::dpct_malloc((void **)&knodesD, knodes_mem);

  //==================================================50
  //  currKnodeD
  //==================================================50

  long *currKnodeD;
  dpct::dpct_malloc((void **)&currKnodeD, count * sizeof(long));

  //==================================================50
  //  offsetD
  //==================================================50

  long *offsetD;
  dpct::dpct_malloc((void **)&offsetD, count * sizeof(long));

  //==================================================50
  //  keysD
  //==================================================50

  int *keysD;
  dpct::dpct_malloc((void **)&keysD, count * sizeof(int));

  //==================================================50
  //  ansD
  //==================================================50

  record *ansD;
  dpct::dpct_malloc((void **)&ansD, count * sizeof(record));

  //==================================================50
  //  recordsD
  //==================================================50

  dpct::async_dpct_memcpy(recordsD, records, records_mem, dpct::host_to_device);

  //==================================================50
  //  knodesD
  //==================================================50

  dpct::async_dpct_memcpy(knodesD, knodes, knodes_mem, dpct::host_to_device);

  //==================================================50
  //  currKnodeD
  //==================================================50

  dpct::async_dpct_memcpy(currKnodeD, currKnode, count * sizeof(long),
                          dpct::host_to_device);

  //==================================================50
  //  offsetD
  //==================================================50

  dpct::async_dpct_memcpy(offsetD, offset, count * sizeof(long),
                          dpct::host_to_device);

  //==================================================50
  //  keysD
  //==================================================50

  dpct::async_dpct_memcpy(keysD, keys, count * sizeof(int),
                          dpct::host_to_device);

  //==================================================50
  //  ansD
  //==================================================50

  dpct::async_dpct_memcpy(ansD, ans, count * sizeof(record),
                          dpct::host_to_device);

  //======================================================================================================================================================150
  // findK kernel
  //======================================================================================================================================================150

  {
    dpct::buffer_t knodesD_buf_ct1 = dpct::get_buffer(knodesD);
    dpct::buffer_t recordsD_buf_ct3 = dpct::get_buffer(recordsD);
    dpct::buffer_t currKnodeD_buf_ct4 = dpct::get_buffer(currKnodeD);
    dpct::buffer_t offsetD_buf_ct5 = dpct::get_buffer(offsetD);
    dpct::buffer_t keysD_buf_ct6 = dpct::get_buffer(keysD);
    dpct::buffer_t ansD_buf_ct7 = dpct::get_buffer(ansD);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto knodesD_acc_ct1 =
          knodesD_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto recordsD_acc_ct3 =
          recordsD_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto currKnodeD_acc_ct4 =
          currKnodeD_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
      auto offsetD_acc_ct5 =
          offsetD_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
      auto keysD_acc_ct6 =
          keysD_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
      auto ansD_acc_ct7 =
          ansD_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                sycl::range<3>(1, 1, threadsPerBlock),
                            sycl::range<3>(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item_ct1) {
            findK(maxheight, (const knode *)(&knodesD_acc_ct1[0]), knodes_elem,
                  (const record *)(&recordsD_acc_ct3[0]),
                  (long *)(&currKnodeD_acc_ct4[0]),
                  (long *)(&offsetD_acc_ct5[0]),
                  (const int *)(&keysD_acc_ct6[0]),
                  (record *)(&ansD_acc_ct7[0]), item_ct1);
          });
    });
  }

  //==================================================50
  //  ansD
  //==================================================50

  dpct::dpct_memcpy(ans, ansD, count * sizeof(record), dpct::device_to_host);

  dpct::dpct_free(recordsD);
  dpct::dpct_free(knodesD);
  dpct::dpct_free(currKnodeD);
  dpct::dpct_free(offsetD);
  dpct::dpct_free(keysD);
  dpct::dpct_free(ansD);

  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("ans[%d] = %d\n", i, ans[i].value);
  printf("\n");
#endif

  printf("Total time:\n");
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000); 
}
