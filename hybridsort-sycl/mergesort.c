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
#include <sycl/sycl.hpp>
#include "mergesort.h"

////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE  256
#define ROW_LENGTH  BLOCKSIZE * 4
#define ROWS    4096

constexpr sycl::access::mode sycl_read       = sycl::access::mode::read;
constexpr sycl::access::mode sycl_write      = sycl::access::mode::write;
constexpr sycl::access::mode sycl_read_write = sycl::access::mode::read_write;
constexpr sycl::access::mode sycl_discard_read_write = sycl::access::mode::discard_read_write;
constexpr sycl::access::mode sycl_discard_write = sycl::access::mode::discard_write;

////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////

// Codeplay 
//  error: no viable conversion from 'vec<typename
//        detail::vec_ops::logical_return<sizeof(float)>::type, 1>'
//              (aka 'vec<int, 1>') to 'bool'
//                b.z() = a.y() >= b.z() ? a.y() : b.z();
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

sycl::float4* runMergeSort(sycl::queue &q, int listsize, int divisions,
    sycl::float4 *d_origList, sycl::float4 *d_resultList,
    int *sizes, int *nullElements,
    unsigned int *origOffsets){

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
  size_t global[] = {blocks*THREADS,1,1};
  size_t grid[3];

  const sycl::property_list props = sycl::property::buffer::use_host_ptr();

  // divided by four 
  sycl::buffer<sycl::float4,1> d_resultList_buff (listsize/4);
  sycl::buffer<sycl::float4,1> d_origList_buff (d_origList, listsize/4, props);
  sycl::buffer<int, 1> d_constStartAddr (startaddr, (divisions+1), props);
  d_origList_buff.set_final_data(nullptr);

  q.submit([&](sycl::handler& cgh) {
      auto input_acc = d_origList_buff.get_access<sycl_read>(cgh);
      auto result_acc = d_resultList_buff.get_access<sycl_write>(cgh);
      cgh.parallel_for<class mergesort_first>(
          sycl::nd_range<1>(sycl::range<1>(global[0]), sycl::range<1>(local[0])),
          [=] (sycl::nd_item<1> item) {
          int gid = item.get_global_id(0);
          if (gid < listsize/4) 
          result_acc[gid] = sortElem(input_acc[gid]);
          });
      });

  int nrElems = 2;

  while(true){
    int floatsperthread = (nrElems*4);
    int threadsPerDiv = (int)ceil(largestSize/(float)floatsperthread);
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
    auto tempList = std::move(d_origList_buff);
    d_origList_buff = std::move(d_resultList_buff);
    d_resultList_buff = std::move(tempList);

    global[0] = grid[0]*local[0];

    q.submit([&](sycl::handler& cgh) {
      auto input_acc = d_origList_buff.get_access<sycl_read>(cgh);
      auto result_acc = d_resultList_buff.get_access<sycl_write>(cgh);
      auto constStartAddr_acc = d_constStartAddr.get_access<sycl_read>(cgh);
      cgh.parallel_for<class mergepass>(
        sycl::nd_range<1>(sycl::range<1>(global[0]), sycl::range<1>(local[0])),
        [=] (sycl::nd_item<1> item) {
        #include "kernel_mergeSortPass.sycl"
      });
    });

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
  global[0] = grid[0]*local[0];
  global[1] = grid[1]*local[1];

  sycl::buffer<unsigned int, 1> finalStartAddr(origOffsets, divisions+1, props);
  sycl::buffer<int, 1> nullElems(nullElements, divisions, props);

  // reinterpreted sycl::buffer
  auto d_orig = d_origList_buff.reinterpret<float>(sycl::range<1>(listsize));
  auto d_res = d_resultList_buff.reinterpret<float>(sycl::range<1>(listsize));

  q.submit([&](sycl::handler& cgh) {
      auto orig_acc = d_res.get_access<sycl_read>(cgh);
      auto result_acc = d_orig.get_access<sycl_write>(cgh);
      auto finalStartAddr_acc = finalStartAddr.get_access<sycl_read>(cgh);
      auto nullElems_acc = nullElems.get_access<sycl_read>(cgh);
      auto constStartAddr_acc = d_constStartAddr.get_access<sycl_read>(cgh);
      cgh.parallel_for<class mergepack>(
          sycl::nd_range<2>(sycl::range<2>(global[1],global[0]),
                            sycl::range<2>(local[1], local[0])), [=] (sycl::nd_item<2> item) {
          int idx = item.get_global_id(1);
          int division = item.get_group(0);
          if((finalStartAddr_acc[division] + idx) < finalStartAddr_acc[division + 1])
            result_acc[finalStartAddr_acc[division] + idx] = 
              orig_acc[constStartAddr_acc[division]*4 + nullElems_acc[division] + idx];
          });
      });

  q.submit([&](sycl::handler& cgh) {
      auto orig_acc = d_origList_buff.get_access<sycl_read>(cgh);
      cgh.copy(orig_acc, d_origList);
      });

  free(startaddr);
  q.wait();

  return d_origList;
}
