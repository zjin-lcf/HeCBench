/*Copyright(c) 2020, The Regents of the University of California, Davis.            */
/*                                                                                  */
/*                                                                                  */
/*Redistribution and use in source and binary forms, with or without modification,  */
/*are permitted provided that the following conditions are met :                    */
/*                                                                                  */
/*1. Redistributions of source code must retain the above copyright notice, this    */
/*list of conditions and the following disclaimer.                                  */
/*2. Redistributions in binary form must reproduce the above copyright notice,      */
/*this list of conditions and the following disclaimer in the documentation         */
/*and / or other materials provided with the distribution.                          */
/*                                                                                  */
/*THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND   */
/*ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED     */
/*WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.*/
/*IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,  */
/*INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT */
/*NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR*/
/*PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, */
/*WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) */
/*ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE        */
/*POSSIBILITY OF SUCH DAMAGE.                                                       */
/************************************************************************************/

#pragma once

#include <cstdint>

namespace GpuBTree {
template <typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
int GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::initBTree(
    uint32_t *&d_root, uint32_t *d_pool, uint32_t *d_count, sycl::queue &stream_id) {
  stream_id.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(32), sycl::range<1>(32)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        kernels::init_btree(d_root, d_pool, d_count, item);
    });
  }).wait();

  return 0;
}

template <typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
int GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::insertKeys(
    uint32_t *&d_root, KeyT *&d_keys, ValueT *&d_values, SizeT &count,
    uint32_t *d_pool, uint32_t *d_count, sycl::queue &stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_BUILD_ - 1) / BLOCKSIZE_BUILD_;

  stream_id.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCKSIZE_BUILD_),
                        sycl::range<1>(BLOCKSIZE_BUILD_)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        kernels::insert_keys(d_root, d_keys, d_values, count, d_pool, d_count, item);
    });
  }).wait();

  return 0;
}

template <typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
int GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::searchKeys(
    uint32_t *&d_root, KeyT *&d_queries, ValueT *&d_results, SizeT &count, uint32_t *d_pool,
    sycl::queue &stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_SEARCH_ - 1) / BLOCKSIZE_SEARCH_;
  stream_id.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCKSIZE_SEARCH_),
                        sycl::range<1>(BLOCKSIZE_SEARCH_)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        kernels::search_b_tree(d_root, d_queries, d_results, count, d_pool, item);
    });
  }).wait();

  return 0;
}

template <typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
int GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::deleteKeys(
    uint32_t *&d_root, KeyT *&d_queries, SizeT &count, uint32_t *d_pool, sycl::queue &stream_id) {
  const uint32_t num_blocks = (count + BLOCKSIZE_SEARCH_ - 1) / BLOCKSIZE_SEARCH_;
  stream_id.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_blocks * BLOCKSIZE_SEARCH_),
                        sycl::range<1>(BLOCKSIZE_SEARCH_)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        kernels::delete_b_tree(d_root, d_queries, count, d_pool, item);
    });
  }).wait();

  return 0;
}

template <typename KeyT, typename ValueT, typename SizeT, typename AllocatorT>
int GpuBTreeMap<KeyT, ValueT, SizeT, AllocatorT>::rangeQuery(
    uint32_t *&d_root, KeyT *&d_queries_lower, KeyT *&d_queries_upper,
    ValueT *&d_range_results, SizeT &count, SizeT &range_lenght,
    uint32_t *d_pool,
    sycl::queue &stream_id) {
  const uint32_t block_size = 256;
  const uint32_t num_blocks = (count + block_size - 1) / block_size;
  stream_id.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size),
                        sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        kernels::range_b_tree(d_root, d_queries_lower, d_queries_upper,
                       d_range_results, count, range_lenght, d_pool, item);
    });
  }).wait();

  return 0;
}
};  // namespace GpuBTree
