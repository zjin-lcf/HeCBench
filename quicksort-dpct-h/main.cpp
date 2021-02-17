/*
   Copyright (c) 2014-2019, Intel Corporation
   Redistribution and use in source and binary forms, with or without 
   modification, are permitted provided that the following conditions 
   are met:
 * Redistributions of source code must retain the above copyright 
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above 
 copyright notice, this list of conditions and the following 
 disclaimer in the documentation and/or other materials provided 
 with the distribution.
 * Neither the name of Intel Corporation nor the names of its 
 contributors may be used to endorse or promote products 
 derived from this software without specific prior written 
 permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 POSSIBILITY OF SUCH DAMAGE.
 */

// QuicksortMain.cpp : Defines the entry point for the console application.
//

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <map>

#define RUN_CPU_SORTS
//#define GET_DETAILED_PERFORMANCE

#define gpucheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(int code, const char *file, int line, bool abort = true)
{
}

// Types:
typedef unsigned int uint;
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

/// return a timestamp with sub-second precision 
/** QueryPerformanceCounter and clock_gettime have an undefined starting point (null/zero)     
 *  and can wrap around, i.e. be nulled again. **/ 
double seconds() { 
  struct timespec now;   
  clock_gettime(CLOCK_MONOTONIC, &now);   
  return now.tv_sec + now.tv_nsec / 1000000000.0; 
}


bool parseArgs(int argc, char** argv, unsigned int* test_iterations, unsigned int* widthReSz, unsigned int* heightReSz)
{  
  const char sUsageString[512] = "Usage: Quicksort [num test iterations] [SurfWidth(^2 only)] [SurfHeight(^2 only)]";

  if (argc != 4)
  {
    printf(sUsageString);
    return false;
  }
  else
  {
    *test_iterations  = atoi (argv[1]);
    *widthReSz  = atoi (argv[2]);
    *heightReSz  = atoi (argv[3]);
    return true;
  }
}


#include "Quicksort.h"
#include "QuicksortKernels.dp.hpp"

template <class T>
T* partition(T* left, T* right, T pivot) {
  // move pivot to the end
  T temp = *right;
  *right = pivot;
  *left = temp;

  T* store = left;

  for(T* p = left; p != right; p++) {
    if (*p < pivot) {
      temp = *store;
      *store = *p;
      *p = temp;
      store++;
    }
  }

  temp = *store;
  *store = pivot;
  *right = temp;

  return store;
}

  template <class T>
void quicksort(T* data, int left, int right)
{
  T* store = partition(data + left, data + right, data[left]);
  int nright = store-data;
  int nleft = nright+1;

  if (left < nright) {
    if (nright - left > 32) {
      quicksort(data, left, nright);
    } else
      std::sort(data + left, data + nright + 1);
  }

  if (nleft < right) {
    if (right - nleft > 32)  {
      quicksort(data, nleft, right); 
    } else {
      std::sort(data + nleft, data + right + 1);
    }
  }
}

template <class T>
void gqsort(T *db, 
    T *dnb, 
    std::vector<block_record<T>>& blocks,
    std::vector<parent_record>& parents,
    std::vector<work_record<T>>& news,
    bool reset) {

  news.resize(blocks.size()*2);

#ifdef GET_DETAILED_PERFORMANCE
  static double absoluteTotal = 0.0;
  static uint count = 0;

  if (reset) {
    absoluteTotal = 0.0;
    count = 0;
  }

  double beginClock, endClock;
  beginClock = seconds();
#endif

  block_record<T> *blocksb;
  parent_record *parentsb;
  work_record<T> *newsb;
  /*
  DPCT1003:4: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((blocksb = (block_record<T> *)dpct::dpct_malloc(
                sizeof(block_record<T>) * blocks.size()),
            0));
  /*
  DPCT1003:5: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((parentsb = (parent_record *)dpct::dpct_malloc(
                sizeof(parent_record) * parents.size()),
            0));
  /*
  DPCT1003:6: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((newsb = (work_record<T> *)dpct::dpct_malloc(sizeof(work_record<T>) *
                                                        news.size()),
            0));
  /*
  DPCT1003:7: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_memcpy(blocksb, blocks.data(),
                              sizeof(block_record<T>) * blocks.size(),
                              dpct::host_to_device),
            0));
  /*
  DPCT1003:8: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_memcpy(parentsb, parents.data(),
                              sizeof(parent_record) * parents.size(),
                              dpct::host_to_device),
            0));
  /*
  DPCT1003:9: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_memcpy(newsb, news.data(),
                              sizeof(work_record<T>) * news.size(),
                              dpct::host_to_device),
            0));

  int blockSize = blocks.size();
  {
    std::pair<dpct::buffer_t, size_t> db_buf_ct0 =
        dpct::get_buffer_and_offset(db);
    size_t db_offset_ct0 = db_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> dnb_buf_ct1 =
        dpct::get_buffer_and_offset(dnb);
    size_t dnb_offset_ct1 = dnb_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> blocksb_buf_ct2 =
        dpct::get_buffer_and_offset(blocksb);
    size_t blocksb_offset_ct2 = blocksb_buf_ct2.second;
    dpct::buffer_t parentsb_buf_ct3 = dpct::get_buffer(parentsb);
    std::pair<dpct::buffer_t, size_t> newsb_buf_ct4 =
        dpct::get_buffer_and_offset(newsb);
    size_t newsb_offset_ct4 = newsb_buf_ct4.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<uint, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          lt_acc_ct1(sycl::range<1>(129 /*GQSORT_LOCAL_WORKGROUP_SIZE+1*/),
                     cgh);
      sycl::accessor<uint, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          gt_acc_ct1(sycl::range<1>(129 /*GQSORT_LOCAL_WORKGROUP_SIZE+1*/),
                     cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          ltsum_acc_ct1(cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          gtsum_acc_ct1(cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          lbeg_acc_ct1(cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          gbeg_acc_ct1(cgh);
      auto db_acc_ct0 =
          db_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto dnb_acc_ct1 =
          dnb_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto blocksb_acc_ct2 =
          blocksb_buf_ct2.first.get_access<sycl::access::mode::read_write>(cgh);
      auto parentsb_acc_ct3 =
          parentsb_buf_ct3.get_access<sycl::access::mode::read_write>(cgh);
      auto newsb_acc_ct4 =
          newsb_buf_ct4.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, blockSize) *
                  sycl::range<3>(1, 1, GQSORT_LOCAL_WORKGROUP_SIZE),
              sycl::range<3>(1, 1, GQSORT_LOCAL_WORKGROUP_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            T *db_ct0 = (T *)(&db_acc_ct0[0] + db_offset_ct0);
            T *dnb_ct1 = (T *)(&dnb_acc_ct1[0] + dnb_offset_ct1);
            block_record<T> *blocksb_ct2 =
                (block_record<T> *)(&blocksb_acc_ct2[0] + blocksb_offset_ct2);
            work_record<T> *newsb_ct4 =
                (work_record<T> *)(&newsb_acc_ct4[0] + newsb_offset_ct4);
            gqsort_kernel(
                db_ct0, dnb_ct1, blocksb_ct2,
                (parent_record *)(&parentsb_acc_ct3[0]), newsb_ct4, item_ct1,
                lt_acc_ct1.get_pointer(), gt_acc_ct1.get_pointer(),
                ltsum_acc_ct1.get_pointer(), gtsum_acc_ct1.get_pointer(),
                lbeg_acc_ct1.get_pointer(), gbeg_acc_ct1.get_pointer());
          });
    });
  }

  /*
  DPCT1010:10: SYCL uses exceptions to report errors and does not use the
   * error codes. The call was replaced with 0. You need to rewrite this code.

   */
  gpucheck(0);
  /*
  DPCT1003:11: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::get_current_device().queues_wait_and_throw(), 0));
  /*
  DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_memcpy(news.data(), newsb,
                              sizeof(work_record<T>) * news.size(),
                              dpct::device_to_host),
            0));

  /*
  DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_free(blocksb), 0));
  /*
  DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_free(parentsb), 0));
  /*
  DPCT1003:15: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_free(newsb), 0));

#ifdef GET_DETAILED_PERFORMANCE
  endClock = seconds();
  double totalTime = endClock - beginClock;
  absoluteTotal += totalTime;
  std::cout << ++count << ": gqsort time " << absoluteTotal * 1000 << " ms" << std::endl;
#endif

#ifdef DEBUG
  printf("\noutput news\n");
  for (int i = 0; i < news.size(); i++) {
    printf("%u %u %u %u\n", news[i].start, news[i].end, news[i].pivot, news[i].direction);
  }
#endif
}

template <class T>
void lqsort(T *db, T *dnb, std::vector<work_record<T>>& done) {

#ifdef GET_DETAILED_PERFORMANCE
  double beginClock, endClock;
  beginClock = seconds();
#endif
  work_record<T>* doneb;
  //std::cout << "done size is " << done.size() << std::endl;

  /*
  DPCT1003:16: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((doneb = (work_record<T> *)dpct::dpct_malloc(sizeof(work_record<T>) *
                                                        done.size()),
            0));
  /*
  DPCT1003:17: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_memcpy(doneb, done.data(),
                              sizeof(work_record<T>) * done.size(),
                              dpct::host_to_device),
            0));

  int doneSize = done.size();
  {
    std::pair<dpct::buffer_t, size_t> db_buf_ct0 =
        dpct::get_buffer_and_offset(db);
    size_t db_offset_ct0 = db_buf_ct0.second;
    std::pair<dpct::buffer_t, size_t> dnb_buf_ct1 =
        dpct::get_buffer_and_offset(dnb);
    size_t dnb_offset_ct1 = dnb_buf_ct1.second;
    std::pair<dpct::buffer_t, size_t> doneb_buf_ct2 =
        dpct::get_buffer_and_offset(doneb);
    size_t doneb_offset_ct2 = doneb_buf_ct2.second;
    dpct::get_default_queue().submit([&](sycl::handler &cgh) {
      sycl::accessor<workstack_record, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          workstack_acc_ct1(
              sycl::range<1>(6 /*QUICKSORT_BLOCK_SIZE/SORT_THRESHOLD*/), cgh);
      sycl::accessor<int, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          workstack_pointer_acc_ct1(cgh);
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          mys_acc_ct1(sycl::range<1>(1728 /*QUICKSORT_BLOCK_SIZE*/), cgh);
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          mysn_acc_ct1(sycl::range<1>(1728 /*QUICKSORT_BLOCK_SIZE*/), cgh);
      sycl::accessor<T, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          temp_acc_ct1(sycl::range<1>(256 /*SORT_THRESHOLD*/), cgh);
      sycl::accessor<T *, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          s_acc_ct1(cgh);
      sycl::accessor<T *, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          sn_acc_ct1(cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          ltsum_acc_ct1(cgh);
      sycl::accessor<uint, 0, sycl::access::mode::read_write,
                     sycl::access::target::local>
          gtsum_acc_ct1(cgh);
      sycl::accessor<uint, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          lt_acc_ct1(sycl::range<1>(129 /*LQSORT_LOCAL_WORKGROUP_SIZE+1*/),
                     cgh);
      sycl::accessor<uint, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          gt_acc_ct1(sycl::range<1>(129 /*LQSORT_LOCAL_WORKGROUP_SIZE+1*/),
                     cgh);
      auto db_acc_ct0 =
          db_buf_ct0.first.get_access<sycl::access::mode::read_write>(cgh);
      auto dnb_acc_ct1 =
          dnb_buf_ct1.first.get_access<sycl::access::mode::read_write>(cgh);
      auto doneb_acc_ct2 =
          doneb_buf_ct2.first.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for(
          sycl::nd_range<3>(
              sycl::range<3>(1, 1, doneSize) *
                  sycl::range<3>(1, 1, LQSORT_LOCAL_WORKGROUP_SIZE),
              sycl::range<3>(1, 1, LQSORT_LOCAL_WORKGROUP_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            T *db_ct0 = (T *)(&db_acc_ct0[0] + db_offset_ct0);
            T *dnb_ct1 = (T *)(&dnb_acc_ct1[0] + dnb_offset_ct1);
            work_record<T> *doneb_ct2 =
                (work_record<T> *)(&doneb_acc_ct2[0] + doneb_offset_ct2);
            lqsort_kernel(
                db_ct0, dnb_ct1, doneb_ct2, item_ct1,
                workstack_acc_ct1.get_pointer(),
                workstack_pointer_acc_ct1.get_pointer(),
                (T *)mys_acc_ct1.get_pointer(), (T *)mysn_acc_ct1.get_pointer(),
                (T *)temp_acc_ct1.get_pointer(), (T **)s_acc_ct1.get_pointer(),
                (T **)sn_acc_ct1.get_pointer(), ltsum_acc_ct1.get_pointer(),
                gtsum_acc_ct1.get_pointer(), lt_acc_ct1.get_pointer(),
                gt_acc_ct1.get_pointer());
          });
    });
  }
  /*
  DPCT1010:18: SYCL uses exceptions to report errors and does not use the
   * error codes. The call was replaced with 0. You need to rewrite this code.

   */
  gpucheck(0);
  /*
  DPCT1003:19: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::get_current_device().queues_wait_and_throw(), 0));

  // Lets do phase 2 pass
  /*
  DPCT1003:20: Migrated API does not return error code. (*, 0) is inserted.
   * You may need to rewrite this code.
  */
  gpucheck((dpct::dpct_free(doneb), 0));
#ifdef GET_DETAILED_PERFORMANCE
  endClock = seconds();
  double totalTime = endClock - beginClock;
  std::cout << "lqsort time " << totalTime * 1000 << " ms" << std::endl;
#endif
}

size_t optp(size_t s, double k, size_t m) {
  return (size_t)sycl::pow<double>(2, floor(log(s * k + m) / log(2.0) + 0.5));
}

template <class T>
void GPUQSort(size_t size, T* d, T* dn)  {

  // allocate buffers
  T *db, *dnb;
  db = (T *)dpct::dpct_malloc(((sizeof(T) * size) / 64 + 1) * 64);
  dpct::dpct_memcpy(db, d, ((sizeof(T) * size) / 64 + 1) * 64,
                    dpct::host_to_device);
  dnb = (T *)dpct::dpct_malloc(((sizeof(T) * size) / 64 + 1) * 64);
  dpct::dpct_memcpy(dnb, dn, ((sizeof(T) * size) / 64 + 1) * 64,
                    dpct::host_to_device);

  const size_t MAXSEQ = optp(size, 0.00009516, 203);
  const size_t MAX_SIZE = 12 * std::max(MAXSEQ, (size_t)QUICKSORT_BLOCK_SIZE);
  //std::cout << "MAXSEQ = " << MAXSEQ << std::endl;
  uint startpivot = median_host(d[0], d[size/2], d[size-1]);
  std::vector<work_record<T>> work, done, news;
  work.reserve(MAX_SIZE);
  done.reserve(MAX_SIZE);
  news.reserve(MAX_SIZE);
  std::vector<parent_record> parent_records;
  parent_records.reserve(MAX_SIZE);
  std::vector<block_record<T>> blocks;
  blocks.reserve(MAX_SIZE);

  work.push_back(work_record<T>(0, size, startpivot, 1));

  bool reset = true;

  while(!work.empty() /*&& work.size() + done.size() < MAXSEQ*/) {
    size_t blocksize = 0;

    for(auto it = work.begin(); it != work.end(); ++it) {
      blocksize += std::max((it->end - it->start) / MAXSEQ, (size_t)1);
    }
    for(auto it = work.begin(); it != work.end(); ++it) {
      uint start = it->start;
      uint end   = it->end;
      uint pivot = it->pivot;
      uint direction = it->direction;
      uint blockcount = (end - start + blocksize - 1)/blocksize;
      parent_record prnt(start, end, start, end, blockcount-1);
      parent_records.push_back(prnt);

      for(uint i = 0; i < blockcount - 1; i++) {
        uint bstart = start + blocksize*i;
        block_record<T> br(bstart, bstart+blocksize, pivot, direction, parent_records.size()-1);
        blocks.push_back(br);
      }
      block_record<T> br(start + blocksize*(blockcount - 1), end, pivot, direction, parent_records.size()-1);
      blocks.push_back(br);
    }
    //std::cout << " blocks = " << blocks.size() << " parent records = " << parent_records.size() << " news = " << news.size() << std::endl;

    gqsort<T>(db, dnb, blocks, parent_records, news, reset);

    reset = false;
    work.clear();
    parent_records.clear();
    blocks.clear();
    for(auto it = news.begin(); it != news.end(); ++it) {
      if (it->direction != EMPTY_RECORD) {
        if (it->end - it->start <= QUICKSORT_BLOCK_SIZE /*size/MAXSEQ*/) {
          if (it->end - it->start > 0)
            done.push_back(*it);
        } else {
          work.push_back(*it);
        }
      }
    }
    news.clear();
  }
  for(auto it = work.begin(); it != work.end(); ++it) {
    if (it->end - it->start > 0)
      done.push_back(*it);
  }

  lqsort<T>(db, dnb, done);

  dpct::dpct_memcpy(d, db, ((sizeof(T) * size) / 64 + 1) * 64,
                    dpct::device_to_host);
  dpct::dpct_free(db);
  dpct::dpct_free(dnb);
}

template <class T>
int test(uint arraySize, unsigned int  NUM_ITERATIONS, 
    const std::string& type_name) 
{
  double totalTime, quickSortTime, stdSortTime;
  double beginClock, endClock;

  printf("\n\n\n--------------------------------------------------------------------\n");
  printf("Allocating array size of %d\n", arraySize);
  T* pArray = (T*)aligned_alloc (4096, ((arraySize*sizeof(T))/64 + 1)*64);
  T* pArrayCopy = (T*)aligned_alloc (4096, ((arraySize*sizeof(T))/64 + 1)*64);
  std::generate(pArray, pArray + arraySize, [](){static T i = 0; return ++i; });
  std::random_shuffle(pArray, pArray + arraySize);

#ifdef RUN_CPU_SORTS
  std::cout << "Sorting the regular way..." << std::endl;
  std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
  std::sort(pArrayCopy, pArrayCopy + arraySize);
  endClock = seconds();
  totalTime = endClock - beginClock;
  std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
  stdSortTime = totalTime;

  std::cout << "quicksort on the cpu: " << std::endl;
  std::copy(pArray, pArray + arraySize, pArrayCopy);

  beginClock = seconds();
  quicksort(pArrayCopy, 0, arraySize-1);
  endClock = seconds();
  totalTime = endClock - beginClock;
  std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
  quickSortTime = totalTime;
#ifdef TRUST_BUT_VERIFY
  {
    std::vector<uint> verify(arraySize);
    std::copy(pArray, pArray + arraySize, verify.begin());

    std::cout << "verifying: ";
    std::sort(verify.begin(), verify.end());
    bool correct = std::equal(verify.begin(), verify.end(), pArrayCopy);
    unsigned int num_discrepancies = 0;
    if (!correct) {
      for(size_t i = 0; i < arraySize; i++) {
        if (verify[i] != pArrayCopy[i]) {
          //std:: cout << "discrepancy at " << i << " " << pArrayCopy[i] << " expected " << verify[i] << std::endl;
          num_discrepancies++;
        }
      }
    }
    std::cout << std::boolalpha << correct << std::endl;
    if (!correct) {
      char y;
      std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
      std::cin >> y;
    }
  }
#endif
#endif // RUN_CPU_SORTS

  std::cout << "Sorting with GPU quicksort: " << std::endl;
  std::vector<uint> original(arraySize);
  std::copy(pArray, pArray + arraySize, original.begin());

  std::vector<double> times;
  times.resize(NUM_ITERATIONS);
  double AverageTime = 0.0;
  uint num_failures = 0;
  for(uint k = 0; k < NUM_ITERATIONS; k++) {
    std::copy(original.begin(), original.end(), pArray);
    std::vector<uint> seqs;
    std::vector<uint> verify(arraySize);
    std::copy(pArray, pArray + arraySize, verify.begin());

    beginClock = seconds();
    GPUQSort(arraySize, pArray, pArrayCopy);
    endClock = seconds();
    totalTime = endClock - beginClock;
    std::cout << "Time to sort: " << totalTime * 1000 << " ms" << std::endl;
    times[k] = totalTime;
    AverageTime += totalTime;
#ifdef TRUST_BUT_VERIFY
    std::cout << "verifying: ";
    std::sort(verify.begin(), verify.end());
    bool correct = std::equal(verify.begin(), verify.end(), pArray);
    unsigned int num_discrepancies = 0;
    if (!correct) {
      for(size_t i = 0; i < arraySize; i++) {
        if (verify[i] != pArray[i]) {
          std:: cout << "discrepancy at " << i << " " << pArray[i] << " expected " << verify[i] << std::endl;
          num_discrepancies++;
        }
      }
    }
    std::cout << std::boolalpha << correct << std::endl;
    if (!correct) {
      std::cout << "num_discrepancies: " << num_discrepancies << std::endl;
      num_failures ++;
    }
#endif
  }
  std::cout << " Number of failures: " << num_failures << " out of " << NUM_ITERATIONS << std::endl;
  AverageTime = AverageTime/NUM_ITERATIONS;
  std::cout << "Average Time: " << AverageTime * 1000 << " ms" << std::endl;
  double stdDev = 0.0, minTime = 1000000.0, maxTime = 0.0;
  for(uint k = 0; k < NUM_ITERATIONS; k++) 
  {
    stdDev += (AverageTime - times[k])*(AverageTime - times[k]);
    minTime = std::min(minTime, times[k]);
    maxTime = std::max(maxTime, times[k]);
  }

  if (NUM_ITERATIONS > 1) {
    stdDev = sqrt(stdDev/(NUM_ITERATIONS - 1));
    std::cout << "Standard Deviation: " << stdDev * 1000 << std::endl;
    std::cout << "%error (3*stdDev)/Average: " << 3*stdDev / AverageTime * 100 << "%" << std::endl;
    std::cout << "min time: " << minTime * 1000 << " ms" << std::endl;
    std::cout << "max time: " << maxTime * 1000 << " ms" << std::endl;
  }

#ifdef RUN_CPU_SORTS
  std::cout << "Average speedup over CPU quicksort: " << quickSortTime/AverageTime << std::endl;
  std::cout << "Average speedup over CPU std::sort: " << stdSortTime/AverageTime << std::endl;
#endif // RUN_CPU_SORTS

  printf("-------done--------------------------------------------------------\n");
  free(pArray);
  free(pArrayCopy);

  return 0;
}


int main(int argc, char** argv)
{
  unsigned int  NUM_ITERATIONS;
  uint      heightReSz, widthReSz;


  bool success = parseArgs (argc, argv, &NUM_ITERATIONS, &widthReSz, &heightReSz);
  if (!success) return -1;
  uint arraySize = widthReSz*heightReSz;
  test<uint>(arraySize, NUM_ITERATIONS, "uint");
  test<float>(arraySize, NUM_ITERATIONS, "float");
  test<double>(arraySize, NUM_ITERATIONS, "double");
  return 0;
}



