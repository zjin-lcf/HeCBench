/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_DECODE_PARAMS_CUH_
#define FLASHINFER_DECODE_PARAMS_CUH_

#include <cuda_runtime.h>
#include <cstdint>
#include "layout.cuh"

namespace flashinfer {

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_>
struct SingleDecodeParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = int32_t;
  DTypeQ* q;
  DTypeKV* k;
  DTypeKV* v;
  DTypeO* o;
  float* lse;
  float* maybe_alibi_slopes;
  uint32_t kv_len;
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t q_stride_n;
  uint32_t q_stride_h;
  uint32_t kv_stride_n;
  uint32_t kv_stride_h;
  int32_t window_left;
  float logits_soft_cap;
  float sm_scale;
  float rope_rcp_scale;
  float rope_rcp_theta;
  uint32_t kv_chunk_size;

  __device__ __host__ SingleDecodeParams()
      : q(nullptr),
        k(nullptr),
        v(nullptr),
        o(nullptr),
        lse(nullptr),
        maybe_alibi_slopes(nullptr),
        kv_len(0),
        num_qo_heads(0),
        num_kv_heads(0),
        q_stride_n(0),
        q_stride_h(0),
        kv_stride_n(0),
        kv_stride_h(0),
        window_left(0),
        logits_soft_cap(0.0f),
        sm_scale(0.0f),
        rope_rcp_scale(0.0f),
        rope_rcp_theta(0.0f),
        kv_chunk_size(0) {}

  __device__ __host__ SingleDecodeParams(DTypeQ* q, DTypeKV* k, DTypeKV* v, DTypeO* o,
                                         float* maybe_alibi_slopes, uint32_t seq_len,
                                         uint32_t num_qo_heads, uint32_t num_kv_heads,
                                         QKVLayout kv_layout, uint32_t head_dim,
                                         int32_t window_left, float logits_soft_cap, float sm_scale,
                                         float rope_scale, float rope_theta)
      : q(q),
        k(k),
        v(v),
        o(o),
        lse(nullptr),
        maybe_alibi_slopes(maybe_alibi_slopes),
        kv_len(seq_len),
        num_qo_heads(num_qo_heads),
        num_kv_heads(num_kv_heads),
        q_stride_n(num_qo_heads * head_dim),
        q_stride_h(head_dim),
        kv_stride_n((kv_layout == QKVLayout::kNHD) ? num_kv_heads * head_dim : head_dim),
        kv_stride_h((kv_layout == QKVLayout::kNHD) ? head_dim : seq_len * head_dim),
        window_left(window_left),
        logits_soft_cap(logits_soft_cap),
        sm_scale(sm_scale),
        rope_rcp_scale(1.f / rope_scale),
        rope_rcp_theta(1.f / rope_theta),
        kv_chunk_size(0) {}

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const { return 1; }

  __host__ __device__ __forceinline__ uint32_t get_kv_len(uint32_t batch_idx) const {
    return kv_len;
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_DECODE_PARAMS_CUH_
