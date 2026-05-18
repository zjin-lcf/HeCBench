/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_POS_ENC_CUH_
#define FLASHINFER_POS_ENC_CUH_

#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

#include "layout.dp.hpp"
#include "math.dp.hpp"
#include "utils.dp.hpp"
#include "vec_dtypes.dp.hpp"

namespace syclex = sycl::ext::oneapi::experimental;
using namespace sycl::info::device;
using namespace sycl::info::kernel_device_specific;

namespace flashinfer {

class kernel_0;
class kernel_1;

/*!
 * \brief An enumeration class that defines different modes for applying RoPE
 *   (Rotary Positional Embeddings).
 */
enum class PosEncodingMode {
  // No rotary positional embeddings
  kNone = 0U,
  // Apply Llama-style rope.
  kRoPELlama = 1U,
  // Apply ALiBi bias
  kALiBi = 2U
};

/*!
 * \brief Convert PosEncodingMode to string
 * \param pos_encoding_mode A PosEncodingMode value
 */
inline std::string PosEncodingModeToString(const PosEncodingMode& pos_encoding_mode) {
  switch (pos_encoding_mode) {
    case PosEncodingMode::kNone:
      return "None";
    case PosEncodingMode::kRoPELlama:
      return "Llama";
    case PosEncodingMode::kALiBi:
      return "ALiBi";
    default:
      return "Unknown";
  }
}

inline float get_alibi_slope(uint32_t head_idx, uint32_t num_heads) {
  int n = math::ptx_exp2((int)math::ptx_log2(num_heads));
  return head_idx < n ? math::ptx_exp2(-8. * float(head_idx + 1) / float(n))
                      : math::ptx_exp2(-4. * float((head_idx + 1 - n) * 2 - 1) / float(n));
}

/*!
 * \brief Apply RoPE (Rotary Positional Embeddings) to x[0: head_dim],
 *   return thread-local vector
 * \tparam vec_size A template integer indicates the vector size used
 *   in the kernel
 * \tparam bdx A template integer indicates the blockDim.x
 * \tparam T A template type indicates the x data type
 * \param x A pointer to the start of x data
 * \param freq A vector of float indicates the thread-local rope frequency
 * \param offset A integer indicates the offset of the position in RoPE
 */
template <uint32_t vec_size, uint32_t bdx, typename T>
inline vec_t<float, vec_size>
vec_apply_llama_rope(const T *x, const vec_t<float, vec_size> &freq,
                     int32_t offset, const sycl::nd_item<3> &item_ct1,
                     const uint32_t rotary_dim = vec_size * bdx) {
  vec_t<float, vec_size> permuted_vec, vec;
  vec.cast_load(x + item_ct1.get_local_id(2) * vec_size);

  if (item_ct1.get_local_id(2) * vec_size < rotary_dim) {
    permuted_vec.cast_load(
        x + ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2)
                 ? item_ct1.get_local_id(2) * vec_size + rotary_dim / 2
                 : item_ct1.get_local_id(2) * vec_size - rotary_dim / 2));
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      float embed = float(offset) * freq[i];
      float cos, sin;
      sin = sycl::sincos(embed, &cos);
          //embed,
          //sycl::address_space_cast<sycl::access::address_space::generic_space,
           //                        sycl::access::decorated::yes>(&cos));
      vec[i] =
          vec[i] * cos + ((item_ct1.get_local_id(2) * vec_size < rotary_dim / 2)
                              ? -permuted_vec[i] : permuted_vec[i]) * sin;
    }
  }
  return vec;
}

}  // namespace flashinfer

#endif  // FLASHINFER_POS_ENC_CUH_
