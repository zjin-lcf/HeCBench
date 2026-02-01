/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include "attention_dtypes.h"
#include <sycl/sycl.hpp>
#include <float.h>
#include <type_traits>

#define VLLM_LDG(arg) *(arg)

#define VLLM_SHFL_XOR_SYNC(var, lane_mask) \
  sycl::permute_group_by_xor(sg, var, lane_mask)

#define VLLM_SHFL_SYNC(var, src_lane)  \
  sycl::select_from_group(sg, var, src_lane)

// Q*K^T operation.
template <int THREAD_GROUP_SIZE, typename Vec, int N>
inline float qk_dot_(const Vec (&q)[N], const Vec (&k)[N], sycl::sub_group &sg) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = vfma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline float dot(const Vec (&q)[N], const Vec (&k)[N], sycl::sub_group &sg) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k, sg);
  }
};
