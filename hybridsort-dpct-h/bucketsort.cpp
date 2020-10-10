#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
#include "bucketsort.h"

// CUDA kernels
#include "kernel_bucketcount.h"
#include "kernel_bucketprefix.h"
#include "kernel_bucketsort.h"
#include "kernel_histogram.h"

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const int histosize = 1024;
  //  ////////////////////////////////////////////////////////////////////////////
  //  // First pass - Create 1024 bin histogram
  //  ////////////////////////////////////////////////////////////////////////////
  unsigned int* h_offsets = (unsigned int *) malloc(DIVISIONS * sizeof(unsigned int));
  for(int i = 0; i < DIVISIONS; i++){
    h_offsets[i] = 0;
  }
  float* pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));
  float* historesult = (float *)malloc(histosize * sizeof(float));

  int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;

  float *d_input_buff;
  unsigned int* d_offsets_buff;
  dpct::dpct_malloc((void **)&d_input_buff,
                    sizeof(float) * (listsize + DIVISIONS * 4));
  dpct::async_dpct_memcpy(d_input_buff, d_input,
                          sizeof(float) * (listsize + DIVISIONS * 4),
                          dpct::host_to_device);
  dpct::dpct_malloc((void **)&d_offsets_buff, sizeof(unsigned int) * DIVISIONS);

  size_t global_histogram = 6144;

#ifdef HISTO_WG_SIZE_0
  size_t local_histogram = HISTO_WG_SIZE_0;
#else
  size_t local_histogram = 96;
#endif

  dpct::async_dpct_memcpy(d_offsets_buff, h_offsets,
                          sizeof(unsigned int) * DIVISIONS,
                          dpct::host_to_device);
  {
    dpct::buffer_t d_offsets_buff_buf_ct0 = dpct::get_buffer(d_offsets_buff);
    dpct::buffer_t d_input_buff_buf_ct1 = dpct::get_buffer(d_input_buff);
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_Hist_acc_ct1(sycl::range<1>(3072 /*HISTOGRAM_BLOCK_MEMORY*/), cgh);
      auto d_offsets_buff_acc_ct0 =
          d_offsets_buff_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_input_buff_acc_ct1 =
          d_input_buff_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, global_histogram / local_histogram) *
                  sycl::range<3>(1, 1, local_histogram),
              sycl::range<3>(1, 1, local_histogram)),
          [=](sycl::nd_item<3> item_ct1) {
            histogram1024((unsigned int *)(&d_offsets_buff_acc_ct0[0]),
                          (const float *)(&d_input_buff_acc_ct1[0]), listsize,
                          minimum, maximum, item_ct1,
                          s_Hist_acc_ct1.get_pointer());
          });
    });
  }
  dpct::dpct_memcpy(h_offsets, d_offsets_buff, sizeof(unsigned int) * histosize,
                    dpct::device_to_host);

  for(int i=0; i<histosize; i++) {
    historesult[i] = (float)h_offsets[i];
  }


  //  ///////////////////////////////////////////////////////////////////////////
  //  // Calculate pivot points (CPU algorithm)
  //  ///////////////////////////////////////////////////////////////////////////
  calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
      minimum, maximum, pivotPoints,
      (maximum - minimum)/(float)histosize);
  //
  //  ///////////////////////////////////////////////////////////////////////////
  //  // Count the bucket sizes in new divisions
  //  ///////////////////////////////////////////////////////////////////////////


  float* d_pivotPoints_buff;
  int* d_indice_buff;
  unsigned int* d_prefixoffsets_buff;

  dpct::dpct_malloc((void **)&d_pivotPoints_buff, sizeof(float) * DIVISIONS);
  dpct::async_dpct_memcpy(d_pivotPoints_buff, pivotPoints,
                          sizeof(float) * DIVISIONS, dpct::host_to_device);

  dpct::dpct_malloc((void **)&d_indice_buff, sizeof(int) * listsize);
  dpct::dpct_malloc((void **)&d_prefixoffsets_buff,
                    sizeof(unsigned int) * blocks * BUCKET_BLOCK_MEMORY);

  {
    dpct::buffer_t d_input_buff_buf_ct0 = dpct::get_buffer(d_input_buff);
    dpct::buffer_t d_indice_buff_buf_ct1 = dpct::get_buffer(d_indice_buff);
    dpct::buffer_t d_prefixoffsets_buff_buf_ct2 =
        dpct::get_buffer(d_prefixoffsets_buff);
    dpct::buffer_t d_pivotPoints_buff_buf_ct3 =
        dpct::get_buffer(d_pivotPoints_buff);
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_offset_acc_ct1(sycl::range<1>(1024 /*BUCKET_BLOCK_MEMORY*/), cgh);
      auto d_input_buff_acc_ct0 =
          d_input_buff_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_indice_buff_acc_ct1 =
          d_indice_buff_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_prefixoffsets_buff_acc_ct2 =
          d_prefixoffsets_buff_buf_ct2
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_pivotPoints_buff_acc_ct3 =
          d_pivotPoints_buff_buf_ct3.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                sycl::range<3>(1, 1, BUCKET_THREAD_N),
                            sycl::range<3>(1, 1, BUCKET_THREAD_N)),
          [=](sycl::nd_item<3> item_ct1) {
            bucketcount((const float *)(&d_input_buff_acc_ct0[0]),
                        (int *)(&d_indice_buff_acc_ct1[0]),
                        (unsigned int *)(&d_prefixoffsets_buff_acc_ct2[0]),
                        (const float *)(&d_pivotPoints_buff_acc_ct3[0]),
                        listsize, item_ct1, s_offset_acc_ct1.get_pointer());
          });
    });
  }

  //
  //  ///////////////////////////////////////////////////////////////////////////
  //  // Prefix scan offsets and align each division to float4 (required by
  //  // mergesort)
  //  ///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
  size_t localpre = BUCKET_WG_SIZE_0;
#else
  size_t localpre = 128;
#endif
  size_t globalpre = DIVISIONS;

  {
    dpct::buffer_t d_prefixoffsets_buff_buf_ct0 =
        dpct::get_buffer(d_prefixoffsets_buff);
    dpct::buffer_t d_offsets_buff_buf_ct1 = dpct::get_buffer(d_offsets_buff);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_prefixoffsets_buff_acc_ct0 =
          d_prefixoffsets_buff_buf_ct0
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_offsets_buff_acc_ct1 =
          d_offsets_buff_buf_ct1.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, globalpre / localpre) *
                                sycl::range<3>(1, 1, localpre),
                            sycl::range<3>(1, 1, localpre)),
          [=](sycl::nd_item<3> item_ct1) {
            bucketprefix((unsigned int *)(&d_prefixoffsets_buff_acc_ct0[0]),
                         (unsigned int *)(&d_offsets_buff_acc_ct1[0]), blocks,
                         item_ct1);
          });
    });
  }

  // copy the sizes from device to host
  dpct::dpct_memcpy(h_offsets, d_offsets_buff, sizeof(unsigned int) * DIVISIONS,
                    dpct::device_to_host);

  origOffsets[0] = 0;
  for(int i=0; i<DIVISIONS; i++){
    origOffsets[i+1] = h_offsets[i] + origOffsets[i];
    if((h_offsets[i] % 4) != 0){
      nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i];
    }
    else nullElements[i] = 0;
  }
  for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4;
  for(int i=0; i<DIVISIONS; i++) {
    if((h_offsets[i] % 4) != 0)  h_offsets[i] = (h_offsets[i] & ~3) + 4;
  }
  for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i];
  for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1];
  h_offsets[0] = 0;

  //  ///////////////////////////////////////////////////////////////////////////
  //  // Finally, sort the lot
  //  ///////////////////////////////////////////////////////////////////////////

  // update the h_offsets on the device
  dpct::async_dpct_memcpy(d_offsets_buff, h_offsets,
                          sizeof(unsigned int) * DIVISIONS,
                          dpct::host_to_device);

  float* d_bucketOutput;
  dpct::dpct_malloc((void **)&d_bucketOutput,
                    sizeof(float) * (listsize + DIVISIONS * 4));
  dpct::async_dpct_memcpy(d_bucketOutput, d_output,
                          sizeof(float) * (listsize + DIVISIONS * 4),
                          dpct::host_to_device);

  size_t localfinal = BUCKET_THREAD_N;
  size_t globalfinal = blocks*BUCKET_THREAD_N;

  {
    dpct::buffer_t d_input_buff_buf_ct0 = dpct::get_buffer(d_input_buff);
    dpct::buffer_t d_indice_buff_buf_ct1 = dpct::get_buffer(d_indice_buff);
    dpct::buffer_t d_bucketOutput_buf_ct2 = dpct::get_buffer(d_bucketOutput);
    dpct::buffer_t d_prefixoffsets_buff_buf_ct3 =
        dpct::get_buffer(d_prefixoffsets_buff);
    dpct::buffer_t d_offsets_buff_buf_ct4 = dpct::get_buffer(d_offsets_buff);
    q_ct1.submit([&](sycl::handler &cgh) {
      sycl::accessor<unsigned int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_offset_acc_ct1(sycl::range<1>(1024 /*BUCKET_BLOCK_MEMORY*/), cgh);
      auto d_input_buff_acc_ct0 =
          d_input_buff_buf_ct0.get_access<sycl::access::mode::read_write>(cgh);
      auto d_indice_buff_acc_ct1 =
          d_indice_buff_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
      auto d_bucketOutput_acc_ct2 =
          d_bucketOutput_buf_ct2.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_prefixoffsets_buff_acc_ct3 =
          d_prefixoffsets_buff_buf_ct3
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_offsets_buff_acc_ct4 =
          d_offsets_buff_buf_ct4.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                sycl::range<3>(1, 1, BUCKET_THREAD_N),
                            sycl::range<3>(1, 1, BUCKET_THREAD_N)),
          [=](sycl::nd_item<3> item_ct1) {
            bucketsort((const float *)(&d_input_buff_acc_ct0[0]),
                       (const int *)(&d_indice_buff_acc_ct1[0]),
                       (float *)(&d_bucketOutput_acc_ct2[0]),
                       (const unsigned int *)(&d_prefixoffsets_buff_acc_ct3[0]),
                       (const unsigned int *)(&d_offsets_buff_acc_ct4[0]),
                       listsize, item_ct1, s_offset_acc_ct1.get_pointer());
          });
    });
  }

  dpct::dpct_memcpy(d_output, d_bucketOutput,
                    sizeof(float) * (listsize + DIVISIONS * 4),
                    dpct::device_to_host);

  dpct::dpct_free(d_bucketOutput);
  dpct::dpct_free(d_input_buff);
  dpct::dpct_free(d_offsets_buff);
  dpct::dpct_free(d_pivotPoints_buff);
  dpct::dpct_free(d_indice_buff);
  dpct::dpct_free(d_prefixoffsets_buff);
  free(pivotPoints);
  free(historesult);
}
////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  float elemsPerSlice = listsize/(float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;
  for(int i=0; i<histosize; i++)
  {
    if(i == histosize - 1){
      if(!(p_idx < divisions)){
        pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      }
      break;
    }
    while(histogram[i] > we_need){
      if(!(p_idx < divisions)){
        printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
        exit(0);
      }
      pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      startsAt += (we_need/histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }
    // grab what we can from what remains of it
    we_need -= histogram[i];

    startsAt = endsAt;
    endsAt += histo_width;
  }
  while(p_idx < divisions){
    pivotPoints[p_idx] = pivotPoints[p_idx-1];
    p_idx++;
  }
}
