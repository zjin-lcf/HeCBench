#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <float.h>
#include "mergesort.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE  256
#define ROW_LENGTH  BLOCKSIZE * 4
#define ROWS    4096

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////

SYCL_EXTERNAL
sycl::float4 sortElem(sycl::float4 r) {
  sycl::float4 nr;

  float xt = r.x();
  float yt = r.y();
  float zt = r.z();
  float wt = r.w();

  float nr_xt = xt > yt ? yt : xt;
  float nr_yt = yt > xt ? yt : xt;
  float nr_zt = zt > wt ? wt : zt;
  float nr_wt = wt > zt ? wt : zt;

  xt = nr_xt > nr_zt ? nr_zt : nr_xt;
  yt = nr_yt > nr_wt ? nr_wt : nr_yt;
  zt = nr_zt > nr_xt ? nr_zt : nr_xt;
  wt = nr_wt > nr_yt ? nr_wt : nr_yt;

  nr.x() = xt;
  nr.y() = yt > zt ? zt : yt;
  nr.z() = zt > yt ? zt : yt;
  nr.w() = wt;
  return nr;
}

void sortElement(sycl::float4 *result, sycl::float4 *input, const int size,
                 sycl::nd_item<3> item_ct1)
{
  int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
            item_ct1.get_local_id(2);
  if (gid < size) result[gid] = sortElem(input[gid]);
}

SYCL_EXTERNAL
sycl::float4 getLowest(sycl::float4 a, sycl::float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  a.x() = ax < bw ? ax : bw;
  a.y() = ay < bz ? ay : bz;
  a.z() = az < by ? az : by;
  a.w() = aw < bx ? aw : bx;
  return a;
}

SYCL_EXTERNAL
sycl::float4 getHighest(sycl::float4 a, sycl::float4 b)
{
  float ax = a.x();
  float ay = a.y();
  float az = a.z();
  float aw = a.w();
  float bx = b.x();
  float by = b.y();
  float bz = b.z();
  float bw = b.w();
  b.x() = aw >= bx ? aw : bx;
  b.y() = az >= by ? az : by;
  b.z() = ay >= bz ? ay : bz;
  b.w() = ax >= bw ? ax : bw;
  return b;
}

// the kernel calls the functions defined above
#include "kernel_mergeSortPass.h"

  void
mergepack ( float* result , 
    const float* orig , 
    const int *constStartAddr,
    const unsigned int *finalStartAddr,
    const unsigned int *nullElems ,
    sycl::nd_item<3> item_ct1)
{

  const int gid = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                  item_ct1.get_local_id(2);
  int division = item_ct1.get_group(1);
  if((finalStartAddr[division] + gid) < finalStartAddr[division + 1])
    result[finalStartAddr[division] + gid] = 
      orig[constStartAddr[division]*4 + nullElems[division] + gid];
}

sycl::float4 *runMergeSort(int listsize, int divisions,
                           sycl::float4 *d_origList, sycl::float4 *d_resultList,
                           int *sizes, int *nullElements,
                           unsigned int *origOffsets) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  int *startaddr = (int *)malloc((divisions + 1)*sizeof(int));
  int largestSize = -1;
  startaddr[0] = 0;
  for(int i=1; i<=divisions; i++)
  {
    startaddr[i] = startaddr[i-1] + sizes[i-1];
    if(sizes[i-1] > largestSize) largestSize = sizes[i-1];
  }
  largestSize *= 4;


#ifdef MERGE_WG_SIZE_0
  const int THREADS = MERGE_WG_SIZE_0;
#else
  const int THREADS = 256;
#endif
  size_t local[] = {THREADS,1,1};
  size_t blocks = ((listsize/4)%THREADS == 0) ? (listsize/4)/THREADS : (listsize/4)/THREADS + 1;
  size_t grid[3];

  // divided by four
  sycl::float4 *d_resultList_buff;
  dpct::dpct_malloc((void **)&d_resultList_buff,
                    sizeof(sycl::float4) * listsize / 4);

  sycl::float4 *d_origList_buff;
  dpct::dpct_malloc((void **)&d_origList_buff,
                    sizeof(sycl::float4) * listsize / 4);
  dpct::async_dpct_memcpy(d_origList_buff, d_origList,
                          sizeof(sycl::float4) * listsize / 4,
                          dpct::host_to_device);

  int* d_constStartAddr;
  dpct::dpct_malloc((void **)&d_constStartAddr, sizeof(int) * (divisions + 1));
  dpct::async_dpct_memcpy(d_constStartAddr, startaddr,
                          sizeof(int) * (divisions + 1), dpct::host_to_device);

  {
    dpct::buffer_t d_resultList_buff_buf_ct0 =
        dpct::get_buffer(d_resultList_buff);
    dpct::buffer_t d_origList_buff_buf_ct1 = dpct::get_buffer(d_origList_buff);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_resultList_buff_acc_ct0 =
          d_resultList_buff_buf_ct0.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_origList_buff_acc_ct1 =
          d_origList_buff_buf_ct1.get_access<sycl::access::mode::read_write>(
              cgh);

      cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                             sycl::range<3>(1, 1, THREADS),
                                         sycl::range<3>(1, 1, THREADS)),
                       [=](sycl::nd_item<3> item_ct1) {
                         sortElement(
                             (sycl::float4 *)(&d_resultList_buff_acc_ct0[0]),
                             (sycl::float4 *)(&d_origList_buff_acc_ct1[0]),
                             listsize / 4, item_ct1);
                       });
    });
  }

  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    int threadsPerDiv = (int)ceil(largestSize / (float)floatsperthread);
    int threadsNeeded = threadsPerDiv * divisions;

#ifdef MERGE_WG_SIZE_1
    local[0] = MERGE_WG_SIZE_1;
#else
    local[0] = 208;
#endif

    grid[0] = ((threadsNeeded%local[0]) == 0) ?
      threadsNeeded/local[0] :
      (threadsNeeded/local[0]) + 1;
    if(grid[0] < 8){
      grid[0] = 8;
      local[0] = ((threadsNeeded%grid[0]) == 0) ?
        threadsNeeded / grid[0] :
        (threadsNeeded / grid[0]) + 1;
    }
    // Swap orig/result list
    sycl::float4 *tempList = d_origList_buff;
    d_origList_buff = d_resultList_buff;
    d_resultList_buff = tempList;

    {
      std::pair<dpct::buffer_t, size_t> d_origList_buff_buf_ct0 =
          dpct::get_buffer_and_offset(d_origList_buff);
      size_t d_origList_buff_offset_ct0 = d_origList_buff_buf_ct0.second;
      std::pair<dpct::buffer_t, size_t> d_resultList_buff_buf_ct1 =
          dpct::get_buffer_and_offset(d_resultList_buff);
      size_t d_resultList_buff_offset_ct1 = d_resultList_buff_buf_ct1.second;
      dpct::buffer_t d_constStartAddr_buf_ct2 =
          dpct::get_buffer(d_constStartAddr);
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_origList_buff_acc_ct0 =
            d_origList_buff_buf_ct0.first
                .get_access<sycl::access::mode::read_write>(cgh);
        auto d_resultList_buff_acc_ct1 =
            d_resultList_buff_buf_ct1.first
                .get_access<sycl::access::mode::read_write>(cgh);
        auto d_constStartAddr_acc_ct2 =
            d_constStartAddr_buf_ct2.get_access<sycl::access::mode::read_write>(
                cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, grid[0]) *
                                  sycl::range<3>(1, 1, local[0]),
                              sycl::range<3>(1, 1, local[0])),
            [=](sycl::nd_item<3> item_ct1) {
              const sycl::float4 *d_origList_buff_ct0 =
                  (const sycl::float4 *)(&d_origList_buff_acc_ct0[0] +
                                         d_origList_buff_offset_ct0);
              sycl::float4 *d_resultList_buff_ct1 =
                  (sycl::float4 *)(&d_resultList_buff_acc_ct1[0] +
                                   d_resultList_buff_offset_ct1);
              mergeSortPass(d_origList_buff_ct0, d_resultList_buff_ct1,
                            (const int *)(&d_constStartAddr_acc_ct2[0]),
                            threadsPerDiv, nrElems, item_ct1);
            });
      });
    }

    nrElems *= 2;
    floatsperthread = (nrElems*4);

    if(threadsPerDiv == 1) break;
  }


#ifdef MERGE_WG_SIZE_0
  local[0] = MERGE_WG_SIZE_0;
#else
  local[0] = 256;
#endif
  grid[0] = ((largestSize%local[0]) == 0) ?  largestSize/local[0] : (largestSize/local[0]) + 1;
  grid[1] = divisions;

  unsigned int *d_finalStartAddr;
  dpct::dpct_malloc((void **)&d_finalStartAddr,
                    sizeof(unsigned int) * (divisions + 1));
  dpct::async_dpct_memcpy(d_finalStartAddr, origOffsets,
                          sizeof(unsigned int) * (divisions + 1),
                          dpct::host_to_device);

  unsigned int *d_nullElements;
  dpct::dpct_malloc((void **)&d_nullElements, sizeof(unsigned int) * divisions);
  dpct::async_dpct_memcpy(d_nullElements, nullElements,
                          sizeof(unsigned int) * divisions,
                          dpct::host_to_device);

  sycl::range<3> grids(grid[0], grid[1], 1);
  sycl::range<3> threads(local[0], local[1], 1);

  {
    std::pair<dpct::buffer_t, size_t> d_origList_buff_buf_ct0 =
        dpct::get_buffer_and_offset((float *)d_origList_buff);
    size_t d_origList_buff_offset_ct0 = d_origList_buff_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> d_resultList_buff_buf_ct1 =
        dpct::get_buffer_and_offset((float *)d_resultList_buff);
    size_t d_resultList_buff_offset_ct1 = d_resultList_buff_buf_ct1.second;
    dpct::buffer_t d_constStartAddr_buf_ct2 =
        dpct::get_buffer(d_constStartAddr);
    dpct::buffer_t d_finalStartAddr_buf_ct3 =
        dpct::get_buffer(d_finalStartAddr);
    dpct::buffer_t d_nullElements_buf_ct4 = dpct::get_buffer(d_nullElements);
    q_ct1.submit([&](sycl::handler &cgh) {
      auto d_origList_buff_acc_ct0 =
          d_origList_buff_buf_ct0.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_resultList_buff_acc_ct1 =
          d_resultList_buff_buf_ct1.first
              .get_access<sycl::access::mode::read_write>(cgh);
      auto d_constStartAddr_acc_ct2 =
          d_constStartAddr_buf_ct2.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_finalStartAddr_acc_ct3 =
          d_finalStartAddr_buf_ct3.get_access<sycl::access::mode::read_write>(
              cgh);
      auto d_nullElements_acc_ct4 =
          d_nullElements_buf_ct4.get_access<sycl::access::mode::read_write>(
              cgh);

      auto dpct_global_range = grids * threads;

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(dpct_global_range.get(2), dpct_global_range.get(1),
                             dpct_global_range.get(0)),
              sycl::range<3>(threads.get(2), threads.get(1), threads.get(0))),
          [=](sycl::nd_item<3> item_ct1) {
            float *d_origList_buff_ct0 = (float *)(&d_origList_buff_acc_ct0[0] +
                                                   d_origList_buff_offset_ct0);
            const float *d_resultList_buff_ct1 =
                (const float *)(&d_resultList_buff_acc_ct1[0] +
                                d_resultList_buff_offset_ct1);
            mergepack(d_origList_buff_ct0, d_resultList_buff_ct1,
                      (const int *)(&d_constStartAddr_acc_ct2[0]),
                      (const unsigned int *)(&d_finalStartAddr_acc_ct3[0]),
                      (const unsigned int *)(&d_nullElements_acc_ct4[0]),
                      item_ct1);
          });
    });
  }

  dpct::dpct_memcpy(d_origList, d_origList_buff,
                    sizeof(sycl::float4) * listsize / 4, dpct::device_to_host);

  dpct::dpct_free(d_resultList_buff);
  dpct::dpct_free(d_origList_buff);
  dpct::dpct_free(d_finalStartAddr);
  dpct::dpct_free(d_constStartAddr);
  dpct::dpct_free(d_nullElements);
  free(startaddr);
  return d_origList;
}
