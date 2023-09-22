#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include "bucketsort.h"

constexpr sycl::access::mode sycl_read       = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write      = sycl::access::mode::write;
constexpr sycl::access::mode sycl_read_write = sycl::access::mode::read_write;

inline unsigned int atomicAdd(unsigned int &val, unsigned int operand) 
{
  auto atm = sycl::atomic_ref<unsigned int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>(val);
  return atm.fetch_add(operand);
}

inline unsigned int atomicAddGlobal(unsigned int &val, unsigned int operand) 
{
  auto atm = sycl::atomic_ref<unsigned int,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>(val);
  return atm.fetch_add(operand);
}


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
void bucketSort(sycl::queue &q, float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{

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

  const sycl::property_list props = sycl::property::buffer::use_host_ptr();
  sycl::buffer<float,1> d_input_buff (d_input, listsize + DIVISIONS*4, props);
  sycl::buffer<unsigned int,1> d_offsets_buff (h_offsets, DIVISIONS , props);
  d_offsets_buff.set_final_data(nullptr);

  size_t global_histogram = 6144;

#ifdef HISTO_WG_SIZE_0
  size_t local_histogram = HISTO_WG_SIZE_0;
#else
  size_t local_histogram = 96;
#endif
  q.submit([&](sycl::handler& cgh) {
    auto histoOutput_acc = d_offsets_buff.get_access<sycl_read_write>(cgh);
    auto histoInput_acc = d_input_buff.get_access<sycl_read>(cgh, sycl::range<1>(listsize));
    sycl::local_accessor <unsigned int, 1> s_Hist (sycl::range<1>(HISTOGRAM_BLOCK_MEMORY), cgh);

    cgh.parallel_for<class histogram1024>(
      sycl::nd_range<1>(sycl::range<1>(global_histogram), sycl::range<1>(local_histogram)),
      [=] (sycl::nd_item<1> item) {
        #include "kernel_histogram.sycl"
    });
  });

  q.submit([&](sycl::handler& cgh) {
    auto histoOutput_acc = d_offsets_buff.get_access<sycl_read>(cgh);
    cgh.copy(histoOutput_acc, h_offsets);
  }).wait();

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


  sycl::buffer<float,1> l_pivotpoints_buff(pivotPoints, DIVISIONS, props);
  sycl::buffer<int,1> d_indice_buff(listsize);
  sycl::buffer<unsigned int,1> d_prefixoffsets_buff(blocks * BUCKET_BLOCK_MEMORY);

  //int blocks =((listsize -1) / (BUCKET_THREAD_N*BUCKET_BAND)) + 1;
  size_t global_count = blocks*BUCKET_THREAD_N;
  size_t local_count = BUCKET_THREAD_N;

  q.submit([&](sycl::handler& cgh) {
    auto input_acc = d_input_buff.get_access<sycl_read>(cgh);
    auto indice_acc = d_indice_buff.get_access<sycl_write>(cgh);
    auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_write>(cgh);
    auto l_pivotpoints_acc = l_pivotpoints_buff.get_access<sycl_read>(cgh);
    sycl::local_accessor <unsigned int, 1> s_offset (sycl::range<1>(BUCKET_BLOCK_MEMORY), cgh);
    cgh.parallel_for<class bucketcount>(
      sycl::nd_range<1>(sycl::range<1>(global_count), sycl::range<1>(local_count)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_bucketcount.sycl"
    });
  });

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

  q.submit([&](sycl::handler& cgh) {
    auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_read_write>(cgh);
    auto d_offsets_acc = d_offsets_buff.get_access<sycl_write>(cgh);
    cgh.parallel_for<class prefix>(
      sycl::nd_range<1>(sycl::range<1>(globalpre), sycl::range<1>(localpre)), [=] (sycl::nd_item<1> item) {
      #include "kernel_bucketprefix.sycl"
    });
  });

  // copy the sizes from device to host
  q.submit([&](sycl::handler& cgh) {
    auto d_offsets_buff_acc = d_offsets_buff.get_access<sycl_read>(cgh);
    cgh.copy(d_offsets_buff_acc, h_offsets);
  }).wait();

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
  q.submit([&](sycl::handler& cgh) {
    auto d_offsets_buff_acc = d_offsets_buff.get_access<sycl_write>(cgh);
    cgh.copy(h_offsets, d_offsets_buff_acc);
  });

  sycl::buffer<float,1> d_bucketOutput(d_output, listsize + DIVISIONS*4, props);

  size_t localfinal = BUCKET_THREAD_N;
  size_t globalfinal = blocks*BUCKET_THREAD_N;

  q.submit([&](sycl::handler& cgh) {
    auto input_acc = d_input_buff.get_access<sycl_read>(cgh);
    auto indice_acc = d_indice_buff.get_access<sycl_read>(cgh);
    auto output_acc = d_bucketOutput.get_access<sycl_write>(cgh);
    auto d_prefixoffsets_acc = d_prefixoffsets_buff.get_access<sycl_read>(cgh);
    auto l_offsets_acc = d_offsets_buff.get_access<sycl_read>(cgh);
    sycl::local_accessor <unsigned int, 1> s_offset (sycl::range<1>(BUCKET_BLOCK_MEMORY), cgh);

    cgh.parallel_for<class bucketsort>(
      sycl::nd_range<1>(sycl::range<1>(globalfinal), sycl::range<1>(localfinal)),
      [=] (sycl::nd_item<1> item) {
      #include "kernel_bucketsort.sycl"
    });
  }).wait();

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
