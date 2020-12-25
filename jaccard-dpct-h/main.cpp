/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <algorithm>

using namespace std; 

#define MAX_KERNEL_THREADS 256

// float or double 
typedef float vtype;
typedef vector<vector<vtype>> matrix; 

template<typename T>

T parallel_prefix_sum(const int n, const int *ind, const T *w,
                      sycl::nd_item<3> item_ct1) 
{

  T sum = 0.0;
  T last;

  int mn = (((n + item_ct1.get_local_range().get(2) - 1) /
             item_ct1.get_local_range().get(2)) *
            item_ct1.get_local_range().get(2)); // n in multiple of blockDim.x
  for (int i = item_ct1.get_local_id(2); i < mn;
       i += item_ct1.get_local_range().get(2)) {
    //All threads (especially the last one) must always participate
    //in the shfl instruction, otherwise their sum will be undefined.
    //So, the loop stopping condition is based on multiple of n in loop increments,
    //so that all threads enter into the loop and inside we make sure we do not
    //read out of bounds memory checking for the actual size n.

    //check if the thread is valid
    bool valid  = i<n;

    //Notice that the last thread is used to propagate the prefix sum.
    //For all the threads, in the first iteration the last is 0, in the following
    //iterations it is the value at the last thread of the previous iterations.

    //get the value of the last thread
    last =
        item_ct1.get_sub_group().shuffle(sum, item_ct1.get_local_range(2) - 1);

    //if you are valid read the value from memory, otherwise set your value to 0
    sum = (valid) ? w[ind[i]] : 0.0;

    //do prefix sum (of size warpSize=blockDim.x =< 32)
    for (int j = 1; j < item_ct1.get_local_range().get(2); j *= 2) {
      T v = item_ct1.get_sub_group().shuffle_up(sum, j);
        if (item_ct1.get_local_id(2) >= j) sum += v;
    }
    //shift by last
    sum += last;
    //notice that no __threadfence or __syncthreads are needed in this implementation
  }
  //get the value of the last thread (to all threads)
  last = item_ct1.get_sub_group().shuffle(sum, item_ct1.get_local_range(2) - 1);

  return last;
}

// Volume of neighboors (*weight_s)
template<bool weighted, typename T>
void 
jaccard_row_sum(const int n, const int *csrPtr, const int *csrInd, const T *w, T *work,
                sycl::nd_item<3> item_ct1) 
{

  for (int row = item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
       row < n;
       row += item_ct1.get_group_range(1) * item_ct1.get_local_range().get(1)) {
    int start = csrPtr[row];
    int end   = csrPtr[row+1];
    int length= end-start;
    //compute row sums 
    if (weighted) {
      T sum = parallel_prefix_sum(length, csrInd + start, w, item_ct1);
        if (item_ct1.get_local_id(2) == 0) work[row] = sum;
    }
    else {
      work[row] = (T)length;
    }
  }
}

// Volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
// Note the number of columns is constrained by the number of rows
template<bool weighted, typename T>
void 
jaccard_is(const int n, const int e, const int *csrPtr, const int *csrInd, 
    const T *v, const T *work, T *weight_i, T *weight_s,
    sycl::nd_item<3> item_ct1) 
{

  for (int row = item_ct1.get_local_id(0) +
                 item_ct1.get_group(0) * item_ct1.get_local_range().get(0);
       row < n;
       row += item_ct1.get_group_range(0) * item_ct1.get_local_range().get(0)) {
    for (int j = csrPtr[row] + item_ct1.get_local_id(1) +
                 item_ct1.get_group(1) * item_ct1.get_local_range().get(1);
         j < csrPtr[row + 1];
         j += item_ct1.get_group_range(1) * item_ct1.get_local_range().get(1)) {
      int col = csrInd[j];
      //find which row has least elements (and call it reference row)
      int Ni = csrPtr[row+1] - csrPtr[row];
      int Nj = csrPtr[col+1] - csrPtr[col];
      int ref= (Ni < Nj) ? row : col;
      int cur= (Ni < Nj) ? col : row;

      //compute new sum weights
      weight_s[j] = work[row] + work[col];

      //compute new intersection weights 
      //search for the element with the same column index in the reference row
      for (int i = csrPtr[ref] + item_ct1.get_local_id(2) +
                   item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
           i < csrPtr[ref + 1]; i += item_ct1.get_group_range(2) *
                                     item_ct1.get_local_range().get(2)) {
        int match  =-1;           
        int ref_col = csrInd[i];
        T ref_val = weighted ? v[ref_col] : (T)1.0;

        //binary search (column indices are sorted within each row)
        int left = csrPtr[cur]; 
        int right= csrPtr[cur+1]-1; 
        while(left <= right){
          int middle = (left+right)>>1; 
          int cur_col= csrInd[middle];
          if (cur_col > ref_col) {
            right=middle-1;
          }
          else if (cur_col < ref_col) {
            left=middle+1;
          }
          else {
            match = middle; 
            break; 
          }
        }            

        //if the element with the same column index in the reference row has been found
        if (match != -1){
          dpct::atomic_fetch_add(&weight_i[j], ref_val);
        }
      }
    }
  }
}

template<bool weighted, typename T>
void 
jaccard_jw(const int e, 
    const T *csrVal, 
    const T gamma, 
    const T *weight_i, 
    const T *weight_s, 
    T *weight_j,
    sycl::nd_item<3> item_ct1) 
{
  for (int j = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
       j < e;
       j += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    T Wi =  weight_i[j];
    T Ws =  weight_s[j];
    weight_j[j] = (gamma*csrVal[j])* (Wi/(Ws-Wi));
  }
}



template <bool weighted, typename T>
void 
fill(const int e, T* w, const T value, sycl::nd_item<3> item_ct1) 
{
  for (int j = item_ct1.get_local_id(2) +
               item_ct1.get_group(2) * item_ct1.get_local_range().get(2);
       j < e;
       j += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    // e.g. w[0] is the weight of a non-zeron element when csr_ind[i] equals 0. 
    // So multiple non-zero elements on different rows of a matrix may share 
    // the same weight value
    w[j] = weighted ? (T)(j+1)/e : value; 
  }
}

template <bool weighted, typename T>
void jaccard_weight (const int iteration, const int n, const int e, 
    int* csr_ptr, int* csr_ind, T* csr_val)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();

  const T gamma = (T)0.46;  // arbitrary

  T *d_weight_i, 
    *d_weight_s, 
    *d_weight_j, 
    *d_work;
  int *d_csrInd;
  int *d_csrPtr;
  T *d_csrVal;

#ifdef DEBUG
  T* weight_i = (T*) malloc (sizeof(T) * e);
  T* weight_s = (T*) malloc (sizeof(T) * e);
  T* work = (T*) malloc (sizeof(T) * n);
#endif
  T* weight_j = (T*) malloc (sizeof(T) * e);

  dpct::dpct_malloc((void **)&d_work, sizeof(T) * n);
  dpct::dpct_malloc((void **)&d_weight_i, sizeof(T) * e);
  dpct::dpct_malloc((void **)&d_weight_s, sizeof(T) * e);
  dpct::dpct_malloc((void **)&d_weight_j, sizeof(T) * e);
  dpct::dpct_malloc((void **)&d_csrVal, sizeof(T) * e);
  dpct::dpct_malloc((void **)&d_csrPtr, sizeof(int) * (n + 1));
  dpct::dpct_malloc((void **)&d_csrInd, sizeof(int) * e);

  dpct::async_dpct_memcpy(d_csrPtr, csr_ptr, sizeof(int) * (n + 1),
                          dpct::host_to_device);
  dpct::async_dpct_memcpy(d_csrInd, csr_ind, sizeof(int) * e,
                          dpct::host_to_device);
  dpct::async_dpct_memcpy(d_csrVal, csr_val, sizeof(T) * e,
                          dpct::host_to_device);

  for (int i = 0; i < iteration; i++) {
    sycl::range<3> nthreads(1, 1, 1),
        nblocks(1, 1, 1); // reuse for multiple kernels

    nthreads[0] = MAX_KERNEL_THREADS;
    nthreads[1] = 1;
    nthreads[2] = 1;
    nblocks[0] = (e + MAX_KERNEL_THREADS - 1) / MAX_KERNEL_THREADS;
    nblocks[1] = 1;
    nblocks[2] = 1;

    {
      std::pair<dpct::buffer_t, size_t> d_weight_j_buf_ct1 =
          dpct::get_buffer_and_offset(d_weight_j);
      size_t d_weight_j_offset_ct1 = d_weight_j_buf_ct1.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_weight_j_acc_ct1 =
            d_weight_j_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = nblocks * nthreads;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(nthreads.get(2), nthreads.get(1),
                                             nthreads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              T *d_weight_j_ct1 =
                  (T *)(&d_weight_j_acc_ct1[0] + d_weight_j_offset_ct1);
              fill<weighted, T>(e, d_weight_j_ct1, (T)1.0, item_ct1);
            });
      });
    }
#ifdef DEBUG
    dpct::dpct_memcpy(weight_j, d_weight_j, sizeof(T) * e,
                      dpct::device_to_host);
    for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
#endif

    // initialize volume of intersections
    {
      std::pair<dpct::buffer_t, size_t> d_weight_i_buf_ct1 =
          dpct::get_buffer_and_offset(d_weight_i);
      size_t d_weight_i_offset_ct1 = d_weight_i_buf_ct1.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_weight_i_acc_ct1 =
            d_weight_i_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = nblocks * nthreads;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(nthreads.get(2), nthreads.get(1),
                                             nthreads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              T *d_weight_i_ct1 =
                  (T *)(&d_weight_i_acc_ct1[0] + d_weight_i_offset_ct1);
              fill<false, T>(e, d_weight_i_ct1, (T)0.0, item_ct1);
            });
      });
    }

    // compute row sum with prefix sum
    const int y = 4;
    nthreads[0] = 64 / y;
    nthreads[1] = y;
    nthreads[2] = 1;
    nblocks[0] = 1;
    nblocks[1] =
        (n + nthreads[1] - 1) / nthreads[1]; // less than MAX CUDA BLOCKs
    nblocks[2] = 1;
    {
      dpct::buffer_t d_csrPtr_buf_ct1 = dpct::get_buffer(d_csrPtr);
      dpct::buffer_t d_csrInd_buf_ct2 = dpct::get_buffer(d_csrInd);
      std::pair<dpct::buffer_t, size_t> d_weight_j_buf_ct3 =
          dpct::get_buffer_and_offset(d_weight_j);
      size_t d_weight_j_offset_ct3 = d_weight_j_buf_ct3.second;
      std::pair<dpct::buffer_t, size_t> d_work_buf_ct4 =
          dpct::get_buffer_and_offset(d_work);
      size_t d_work_offset_ct4 = d_work_buf_ct4.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_csrPtr_acc_ct1 =
            d_csrPtr_buf_ct1.get_access<sycl::access::mode::read_write>(cgh);
        auto d_csrInd_acc_ct2 =
            d_csrInd_buf_ct2.get_access<sycl::access::mode::read_write>(cgh);
        auto d_weight_j_acc_ct3 =
            d_weight_j_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_work_acc_ct4 =
            d_work_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = nblocks * nthreads;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(nthreads.get(2), nthreads.get(1),
                                             nthreads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              T *d_weight_j_ct3 =
                  (T *)(&d_weight_j_acc_ct3[0] + d_weight_j_offset_ct3);
              T *d_work_ct4 = (T *)(&d_work_acc_ct4[0] + d_work_offset_ct4);
              jaccard_row_sum<weighted, T>(n, (int *)(&d_csrPtr_acc_ct1[0]),
                                           (int *)(&d_csrInd_acc_ct2[0]),
                                           d_weight_j_ct3, d_work_ct4,
                                           item_ct1);
            });
      });
    }

#ifdef DEBUG
    dpct::dpct_memcpy(work, d_work, sizeof(T) * n, dpct::device_to_host);
    for (int i = 0; i < n; i++) printf("work: %d %f\n", i, work[i]);
#endif

    // compute volume of intersections (*weight_i) and cumulated volume of neighboors (*weight_s)
    // nthreads.x * nthreads.y * nthreads.z <= 256
    nthreads[0] = 32 / y;
    nthreads[1] = y;
    nthreads[2] = 8;
    nblocks[0] = 1;
    nblocks[1] = 1;
    nblocks[2] =
        (n + nthreads[2] - 1) / nthreads[2]; // less than CUDA_MAX_BLOCKS);
    {
      std::pair<dpct::buffer_t, size_t> d_csrPtr_buf_ct2 =
          dpct::get_buffer_and_offset(d_csrPtr);
      size_t d_csrPtr_offset_ct2 = d_csrPtr_buf_ct2.second;
      std::pair<dpct::buffer_t, size_t> d_csrInd_buf_ct3 =
          dpct::get_buffer_and_offset(d_csrInd);
      size_t d_csrInd_offset_ct3 = d_csrInd_buf_ct3.second;
      std::pair<dpct::buffer_t, size_t> d_weight_j_buf_ct4 =
          dpct::get_buffer_and_offset(d_weight_j);
      size_t d_weight_j_offset_ct4 = d_weight_j_buf_ct4.second;
      std::pair<dpct::buffer_t, size_t> d_work_buf_ct5 =
          dpct::get_buffer_and_offset(d_work);
      size_t d_work_offset_ct5 = d_work_buf_ct5.second;
      std::pair<dpct::buffer_t, size_t> d_weight_i_buf_ct6 =
          dpct::get_buffer_and_offset(d_weight_i);
      size_t d_weight_i_offset_ct6 = d_weight_i_buf_ct6.second;
      std::pair<dpct::buffer_t, size_t> d_weight_s_buf_ct7 =
          dpct::get_buffer_and_offset(d_weight_s);
      size_t d_weight_s_offset_ct7 = d_weight_s_buf_ct7.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_csrPtr_acc_ct2 =
            d_csrPtr_buf_ct2.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_csrInd_acc_ct3 =
            d_csrInd_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_j_acc_ct4 =
            d_weight_j_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_work_acc_ct5 =
            d_work_buf_ct5.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_i_acc_ct6 =
            d_weight_i_buf_ct6.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_s_acc_ct7 =
            d_weight_s_buf_ct7.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = nblocks * nthreads;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(nthreads.get(2), nthreads.get(1),
                                             nthreads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              int *d_csrPtr_ct2 =
                  (int *)(&d_csrPtr_acc_ct2[0] + d_csrPtr_offset_ct2);
              int *d_csrInd_ct3 =
                  (int *)(&d_csrInd_acc_ct3[0] + d_csrInd_offset_ct3);
              T *d_weight_j_ct4 =
                  (T *)(&d_weight_j_acc_ct4[0] + d_weight_j_offset_ct4);
              T *d_work_ct5 = (T *)(&d_work_acc_ct5[0] + d_work_offset_ct5);
              T *d_weight_i_ct6 =
                  (T *)(&d_weight_i_acc_ct6[0] + d_weight_i_offset_ct6);
              T *d_weight_s_ct7 =
                  (T *)(&d_weight_s_acc_ct7[0] + d_weight_s_offset_ct7);
              jaccard_is<weighted, T>(n, e, d_csrPtr_ct2, d_csrInd_ct3,
                                      d_weight_j_ct4, d_work_ct5,
                                      d_weight_i_ct6, d_weight_s_ct7, item_ct1);
            });
      });
    }

#ifdef DEBUG
    dpct::dpct_memcpy(weight_i, d_weight_i, sizeof(T) * e,
                      dpct::device_to_host);
    dpct::dpct_memcpy(weight_s, d_weight_s, sizeof(T) * e,
                      dpct::device_to_host);
    for (int i = 0; i < e; i++) printf("wi: %d %f\n", i, weight_i[i]);
    for (int i = 0; i < e; i++) printf("ws: %d %f\n", i, weight_s[i]);
#endif

    // compute jaccard weights
    nthreads[0] = std::min(e, MAX_KERNEL_THREADS);
    nthreads[1] = 1;
    nthreads[2] = 1;
    nblocks[0] =
        (e + nthreads[0] - 1) / nthreads[0]; // less than MAX CUDA BLOCKs
    nblocks[1] = 1;
    nblocks[2] = 1;
    {
      std::pair<dpct::buffer_t, size_t> d_csrVal_buf_ct1 =
          dpct::get_buffer_and_offset(d_csrVal);
      size_t d_csrVal_offset_ct1 = d_csrVal_buf_ct1.second;
      std::pair<dpct::buffer_t, size_t> d_weight_i_buf_ct3 =
          dpct::get_buffer_and_offset(d_weight_i);
      size_t d_weight_i_offset_ct3 = d_weight_i_buf_ct3.second;
      std::pair<dpct::buffer_t, size_t> d_weight_s_buf_ct4 =
          dpct::get_buffer_and_offset(d_weight_s);
      size_t d_weight_s_offset_ct4 = d_weight_s_buf_ct4.second;
      std::pair<dpct::buffer_t, size_t> d_weight_j_buf_ct5 =
          dpct::get_buffer_and_offset(d_weight_j);
      size_t d_weight_j_offset_ct5 = d_weight_j_buf_ct5.second;
      q_ct1.submit([&](sycl::handler &cgh) {
        auto d_csrVal_acc_ct1 =
            d_csrVal_buf_ct1.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_i_acc_ct3 =
            d_weight_i_buf_ct3.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_s_acc_ct4 =
            d_weight_s_buf_ct4.first.get_access<sycl::access::mode::read_write>(
                cgh);
        auto d_weight_j_acc_ct5 =
            d_weight_j_buf_ct5.first.get_access<sycl::access::mode::read_write>(
                cgh);

        auto dpct_global_range = nblocks * nthreads;

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(dpct_global_range.get(2),
                                             dpct_global_range.get(1),
                                             dpct_global_range.get(0)),
                              sycl::range<3>(nthreads.get(2), nthreads.get(1),
                                             nthreads.get(0))),
            [=](sycl::nd_item<3> item_ct1) {
              T *d_csrVal_ct1 =
                  (T *)(&d_csrVal_acc_ct1[0] + d_csrVal_offset_ct1);
              T *d_weight_i_ct3 =
                  (T *)(&d_weight_i_acc_ct3[0] + d_weight_i_offset_ct3);
              T *d_weight_s_ct4 =
                  (T *)(&d_weight_s_acc_ct4[0] + d_weight_s_offset_ct4);
              T *d_weight_j_ct5 =
                  (T *)(&d_weight_j_acc_ct5[0] + d_weight_j_offset_ct5);
              jaccard_jw<weighted, T>(e, d_csrVal_ct1, gamma, d_weight_i_ct3,
                                      d_weight_s_ct4, d_weight_j_ct5, item_ct1);
            });
      });
    }
  }

  dpct::dpct_memcpy(weight_j, d_weight_j, sizeof(T) * e, dpct::device_to_host);
#ifdef DEBUG
  // verify using known values when weighted is true
  float error; 

  if (weighted)
    error =
        std::fabs(weight_j[0] - 0.306667) + std::fabs(weight_j[1] - 0.000000) +
        std::fabs(weight_j[2] - 3.680000) + std::fabs(weight_j[3] - 1.380000) +
        std::fabs(weight_j[4] - 0.788571) + std::fabs(weight_j[5] - 0.460000);

  else
    error =
        std::fabs(weight_j[0] - 0.230000) + std::fabs(weight_j[1] - 0.000000) +
        std::fabs(weight_j[2] - 3.680000) + std::fabs(weight_j[3] - 1.380000) +
        std::fabs(weight_j[4] - 0.920000) + std::fabs(weight_j[5] - 0.460000);

  if (error > 1e-5) {
    for (int i = 0; i < e; i++) printf("wj: %d %f\n", i, weight_j[i]);
    printf("FAILED");
  } else {
    printf("PASSED");
  }
  printf("\n");
#endif

  dpct::dpct_free(d_work);
  dpct::dpct_free(d_weight_i);
  dpct::dpct_free(d_weight_s);
  dpct::dpct_free(d_weight_j);
  dpct::dpct_free(d_csrInd);
  dpct::dpct_free(d_csrVal);
  dpct::dpct_free(d_csrPtr);
  free(weight_j);
#ifdef DEBUG
  free(weight_i);
  free(weight_s);
  free(work);
#endif
}

// Utilities
void printMatrix(const matrix& M) 
{ 
  int m = M.size(); 
  int n = M[0].size(); 
  for (int i = 0; i < m; i++) { 
    for (int j = 0; j < n; j++) 
      cout << M[i][j] << " ";     
    cout << endl; 
  } 
} 

  template <typename T>
void printVector(const vector<T>& V, char* msg) 
{ 
  cout << msg << "[ "; 
  for_each(V.begin(), V.end(), [](int a) { cout << a << " "; }); 
  cout << "]" << endl; 
} 

// Reference: https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/
int main(int argc, char** argv) 
{ 
  int iteration = 10;

#ifdef DEBUG
  matrix M  = { 
    { 0, 0, 0, 1}, 
    { 5, 8, 0, 0}, 
    { 0, 0, 3, 0}, 
    { 0, 6, 0, 1} 
  }; 
#else

  int numRow = atoi(argv[1]);
  int numCol = atoi(argv[2]);
  iteration = atoi(argv[3]);

  srand(2);

  matrix M;
  vector<vtype> rowElems(numCol);
  for (int r = 0; r < numRow; r++) {
    for (int c = 0; c < numCol; c++)
      rowElems[c] = rand() % 10;
    M.push_back(rowElems);
  }
#endif

  int row = M.size();
  int col = M[0].size();
  printf("Number of matrix rows and cols: %d %d\n", row, col);
  vector<vtype> csr_val;
  vector<int> csr_ptr = { 0 }; // require -std=c++11  
  vector<int> csr_ind;
  int nnz = 0; // count Number of non-zero elements in each row

  for (int i = 0; i < row; i++) { 
    for (int j = 0; j < col; j++) { 
      if (M[i][j] != (vtype)0) { 
        csr_val.push_back(M[i][j]); 
        csr_ind.push_back(j); 
        nnz++; 
      } 
    } 
    csr_ptr.push_back(nnz); 
  } 

  // print when the matrix is small
  if (row <= 16 && col <= 16) {
    printMatrix(M); 
    printVector(csr_val, (char*)"values = "); 
    printVector(csr_ptr, (char*)"row pointer = "); 
    printVector(csr_ind, (char*)"col indices = "); 
  }

  jaccard_weight<true, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());
  jaccard_weight<false, vtype>(iteration, row, nnz, csr_ptr.data(), csr_ind.data(), csr_val.data());

  return 0; 
} 

