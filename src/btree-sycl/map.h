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

#include <stdio.h>

#include <cstdint>

namespace GpuBTree {

template<typename KeyT,
         typename ValueT,
         typename SizeT,
         typename AllocatorT>
class GpuBTreeMap {
 private:
  static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
  static constexpr uint32_t DELETED_KEY = 0xFFFFFFFF;
  static constexpr uint32_t BLOCKSIZE_BUILD_ = 256;
  static constexpr uint32_t BLOCKSIZE_SEARCH_ = 256;

  SizeT _num_keys;
  int _device_id;
  uint32_t* _d_root;
  AllocatorT _mem_allocator;

  int initBTree(uint32_t *&root, uint32_t *d_pool, uint32_t *d_count, sycl::queue &stream_id);

  int insertKeys(uint32_t *&root, KeyT *&d_keys, ValueT *&d_values,
                 SizeT &count, uint32_t *d_pool, uint32_t *d_count, sycl::queue &stream_id);

  int searchKeys(uint32_t *&root, KeyT *&d_queries, ValueT *&d_results,
                 SizeT &count, uint32_t *d_pool, sycl::queue &stream_id);

  int deleteKeys(uint32_t *&root, KeyT *&d_queries, SizeT &count,
                 uint32_t *d_pool, sycl::queue &stream_id);

  int rangeQuery(uint32_t *&root, KeyT *&d_queries_lower,
                 KeyT *&d_queries_upper, ValueT *&d_range_results, SizeT &count,
                 SizeT &range_lenght, uint32_t *d_pool, sycl::queue &stream_id);
  bool _handle_memory;
  sycl::queue &m_q;

 public:
  //TODO:  device_id is not used
  GpuBTreeMap(sycl::queue &q, AllocatorT *mem_allocator = nullptr, int device_id = 0) :
   m_q(q)
  {
    if (mem_allocator) {
      _mem_allocator = *mem_allocator;
      _handle_memory = false;
    } else {
      PoolAllocator allocator;
      _mem_allocator = allocator;
      _mem_allocator.init(q);
      memoryUtil::deviceAlloc(q, _d_root, 1);
      _handle_memory = true;
    }
    _device_id = device_id;
    initBTree(_d_root, _mem_allocator.getPool(), _mem_allocator.getCount(), m_q);
  }

  int init(AllocatorT mem_allocator, uint32_t *root_, int deviceId = 0) {
    _device_id = deviceId;
    _mem_allocator = mem_allocator;
    _d_root = root_;  // assumes that the root already contains a one
    return 0;
  }
  ~GpuBTreeMap() {}
  void free() {
    if (_handle_memory) {
      m_q.wait();
      _mem_allocator.free(m_q);
    }
  }

  AllocatorT* getAllocator() { return &_mem_allocator; }

  uint32_t* getRoot() { return _d_root; }

  int insertKeys(KeyT *keys, ValueT *values, SizeT count,
                 SourceT source = SourceT::DEVICE) {
    KeyT* d_keys;
    ValueT* d_values;
    if (source == SourceT::HOST) {
      memoryUtil::deviceAlloc(m_q, d_keys, count);
      memoryUtil::deviceAlloc(m_q, d_values, count);
      memoryUtil::cpyToDevice(m_q, keys, d_keys, count);
      memoryUtil::cpyToDevice(m_q, values, d_values, count);
    } else {
      d_keys = keys;
      d_values = values;
    }

    insertKeys(_d_root, d_keys, d_values, count, 
               _mem_allocator.getPool(), _mem_allocator.getCount(), m_q);

    if (source == SourceT::HOST) {
      memoryUtil::deviceFree(m_q, d_keys);
      memoryUtil::deviceFree(m_q, d_values);
    }

    return 0;
  }

  int searchKeys(KeyT *queries, ValueT *results, SizeT count,
                 SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    ValueT* d_results;
    if (source == SourceT::HOST) {
      memoryUtil::deviceAlloc(m_q, d_queries, count);
      memoryUtil::deviceAlloc(m_q, d_results, count);

      memoryUtil::cpyToDevice(m_q, queries, d_queries, count);
    } else {
      d_queries = queries;
      d_results = results;
    }

    searchKeys(_d_root, d_queries, d_results, count, _mem_allocator.getPool(), m_q);

    if (source == SourceT::HOST) {
      memoryUtil::cpyToHost(m_q, d_results, results, count);
      memoryUtil::deviceFree(m_q, d_queries);
      memoryUtil::deviceFree(m_q, d_results);
    }

    return 0;
  }

  int deleteKeys(KeyT *queries, SizeT count, SourceT source = SourceT::DEVICE) {
    KeyT* d_queries;
    if (source == SourceT::HOST) {
      memoryUtil::deviceAlloc(m_q, d_queries, count);
      memoryUtil::cpyToDevice(m_q, queries, d_queries, count);
    } else {
      d_queries = queries;
    }

    deleteKeys(_d_root, d_queries, count, _mem_allocator.getPool(), m_q);

    if (source == SourceT::HOST) {
      memoryUtil::deviceFree(m_q, d_queries);
    }

    return 0;
  }
  int rangeQuery(KeyT *queries_lower, KeyT *queries_upper, ValueT *results,
                 SizeT average_length, SizeT count,
                 SourceT source = SourceT::DEVICE) {
    KeyT* d_queries_lower;
    KeyT* d_queries_upper;
    KeyT* d_results;
    auto total_range_lenght = count * average_length * 2;
    if (source == SourceT::HOST) {
      memoryUtil::deviceAlloc(m_q, d_queries_lower, count);
      memoryUtil::deviceAlloc(m_q, d_queries_upper, count);
      memoryUtil::deviceAlloc(m_q, d_results, total_range_lenght);
      memoryUtil::cpyToDevice(m_q, queries_lower, d_queries_lower, count);
      memoryUtil::cpyToDevice(m_q, queries_upper, d_queries_upper, count);
    } else {
      d_queries_lower = queries_lower;
      d_queries_upper = queries_upper;
      d_results = results;
    }

    rangeQuery(
        _d_root, d_queries_lower, d_queries_upper, d_results, count,
        average_length, _mem_allocator.getPool(), m_q);

    if (source == SourceT::HOST) {
      memoryUtil::cpyToHost(m_q, d_results, results, total_range_lenght);
      memoryUtil::deviceFree(m_q, d_results);
      memoryUtil::deviceFree(m_q, d_queries_lower);
      memoryUtil::deviceFree(m_q, d_queries_upper);
    }

    return 0;
  }

  double compute_usage() { return _mem_allocator.compute_usage(m_q); }
};
};  // namespace GpuBTree
