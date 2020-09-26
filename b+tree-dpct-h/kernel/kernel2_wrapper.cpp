#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../common.h"
#include "../util/timer/timer.h"
#include "./kernel2.cpp"
#include "./kernel2_wrapper.h"

void 
kernel2_wrapper(  knode *knodes,
    long knodes_elem,
    long knodes_mem,

    int order,
    long maxheight,
    int count,

    long *currKnode,
    long *offset,
    long *lastKnode,
    long *offset_2,
    int *start,
    int *end,
    int *recstart,
    int *reclength)
{

  long long offload_start = get_time();

  //====================================================================================================100
  //  EXECUTION PARAMETERS
  //====================================================================================================100

  int numBlocks;
  numBlocks = count;
  int threadsPerBlock;
  threadsPerBlock = order < 256 ? order : 256;

  printf("# of blocks = %d, # of threads/block = %d (ensure that device can handle)\n", numBlocks, threadsPerBlock);


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
  //  lastKnodeD
  //==================================================50

  long *lastKnodeD;
  dpct::dpct_malloc((void **)&lastKnodeD, count * sizeof(long));

  //==================================================50
  //  offset_2D
  //==================================================50

  long *offset_2D;
  dpct::dpct_malloc((void **)&offset_2D, count * sizeof(long));

  //==================================================50
  //  startD
  //==================================================50

  int *startD;
  dpct::dpct_malloc((void **)&startD, count * sizeof(int));

  //==================================================50
  //  endD
  //==================================================50

  int *endD;
  dpct::dpct_malloc((void **)&endD, count * sizeof(int));

  //==================================================50
  //  ansDStart
  //==================================================50

  int *ansDStart;
  dpct::dpct_malloc((void **)&ansDStart, count * sizeof(int));

  //==================================================50
  //  ansDLength
  //==================================================50

  int *ansDLength;
  dpct::dpct_malloc((void **)&ansDLength, count * sizeof(int));

  dpct::async_dpct_memcpy(knodesD, knodes, knodes_mem, dpct::host_to_device);

  dpct::async_dpct_memcpy(currKnodeD, currKnode, count * sizeof(long),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(offsetD, offset, count * sizeof(long),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(lastKnodeD, lastKnode, count * sizeof(long),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(offset_2D, offset_2, count * sizeof(long),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(startD, start, count * sizeof(int),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(endD, end, count * sizeof(int), dpct::host_to_device);

  dpct::async_dpct_memcpy(ansDStart, recstart, count * sizeof(int),
                          dpct::host_to_device);

  dpct::async_dpct_memcpy(ansDLength, reclength, count * sizeof(int),
                          dpct::host_to_device);

  // [GPU] findRangeK kernel
  {
    dpct::buffer_t knodesD_buf_ct1 = dpct::get_buffer(knodesD);
    dpct::buffer_t currKnodeD_buf_ct3 = dpct::get_buffer(currKnodeD);
    dpct::buffer_t offsetD_buf_ct4 = dpct::get_buffer(offsetD);
    dpct::buffer_t lastKnodeD_buf_ct5 = dpct::get_buffer(lastKnodeD);
    dpct::buffer_t offset_2D_buf_ct6 = dpct::get_buffer(offset_2D);
    dpct::buffer_t startD_buf_ct7 = dpct::get_buffer(startD);
    dpct::buffer_t endD_buf_ct8 = dpct::get_buffer(endD);
    dpct::buffer_t ansDStart_buf_ct9 = dpct::get_buffer(ansDStart);
    dpct::buffer_t ansDLength_buf_ct10 = dpct::get_buffer(ansDLength);
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      auto knodesD_acc_ct1 =
          knodesD_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto currKnodeD_acc_ct3 =
          currKnodeD_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto offsetD_acc_ct4 =
          offsetD_buf_ct4.get_access<sycl::access::mode::read_write>(cgh);
      auto lastKnodeD_acc_ct5 =
          lastKnodeD_buf_ct5.get_access<sycl::access::mode::read_write>(cgh);
      auto offset_2D_acc_ct6 =
          offset_2D_buf_ct6.get_access<sycl::access::mode::read_write>(cgh);
      auto startD_acc_ct7 =
          startD_buf_ct7.get_access<sycl::access::mode::read_write>(cgh);
      auto endD_acc_ct8 =
          endD_buf_ct8.get_access<sycl::access::mode::read_write>(cgh);
      auto ansDStart_acc_ct9 =
          ansDStart_buf_ct9.get_access<sycl::access::mode::read_write>(cgh);
      auto ansDLength_acc_ct10 =
          ansDLength_buf_ct10.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, numBlocks) *
                                sycl::range<3>(1, 1, threadsPerBlock),
                            sycl::range<3>(1, 1, threadsPerBlock)),
          [=](sycl::nd_item<3> item_ct1) {
            findRangeK(
                maxheight, (const knode *)(&knodesD_acc_ct1[0]), knodes_elem,
                (long *)(&currKnodeD_acc_ct3[0]), (long *)(&offsetD_acc_ct4[0]),
                (long *)(&lastKnodeD_acc_ct5[0]),
                (long *)(&offset_2D_acc_ct6[0]),
                (const int *)(&startD_acc_ct7[0]),
                (const int *)(&endD_acc_ct8[0]), (int *)(&ansDStart_acc_ct9[0]),
                (int *)(&ansDLength_acc_ct10[0]), item_ct1);
          });
    });
  }

  dpct::async_dpct_memcpy(recstart, ansDStart, count * sizeof(int),
                          dpct::device_to_host);

  dpct::async_dpct_memcpy(reclength, ansDLength, count * sizeof(int),
                          dpct::device_to_host);

  dpct::get_current_device().queues_wait_and_throw();

  dpct::dpct_free(knodesD);
  dpct::dpct_free(currKnodeD);
  dpct::dpct_free(offsetD);
  dpct::dpct_free(lastKnodeD);
  dpct::dpct_free(offset_2D);
  dpct::dpct_free(startD);
  dpct::dpct_free(endD);
  dpct::dpct_free(ansDStart);
  dpct::dpct_free(ansDLength);

  long long offload_end = get_time();

#ifdef DEBUG
  for (int i = 0; i < count; i++)
    printf("recstart[%d] = %d\n", i, recstart[i]);
  for (int i = 0; i < count; i++)
    printf("reclength[%d] = %d\n", i, reclength[i]);
#endif


  printf("Total time:\n"); 
  printf("%.12f s\n", (float) (offload_end-offload_start) / 1000000); 
}

