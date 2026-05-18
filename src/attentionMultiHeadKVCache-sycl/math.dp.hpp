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
#ifndef FLASHINFER_MATH_CPP_
#define FLASHINFER_MATH_CPP_

#include <sycl/sycl.hpp>

#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
inline float ptx_exp2(float x) {
  float y;
  y = sycl::exp2(x);
  return y;
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
inline float ptx_log2(float x) {
  float y;
  y = sycl::log2(x);
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
inline sycl::half2 ptx_exp2(sycl::half2 x) {
  return {sycl::exp2(x.x()), sycl::exp2(x.y())};
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
inline sycl::half ptx_exp2(sycl::half x) {
  return sycl::exp2(x);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
inline float ptx_rcp(float x) {
  float y;
  y = 1.f / x;
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
inline float shfl_xor_sync(float x, int lane_mask,
                           const sycl::nd_item<3> &item_ct1) {
  return sycl::permute_group_by_xor(item_ct1.get_sub_group(), x, lane_mask);
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
inline sycl::half2 shfl_xor_sync(sycl::half2 x, int lane_mask,
                                 const sycl::nd_item<3> &item_ct1) {
  return sycl::permute_group_by_xor(item_ct1.get_sub_group(), x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
inline float rsqrt(float x) {
  float y;
  y = sycl::rsqrt(x);
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
inline float tanh(float x) {
  float y;
  y = sycl::tanh(x);
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
inline sycl::half2 tanh(sycl::half2 x) {
  return {sycl::tanh(x.x()), sycl::tanh(x.y())};
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
inline sycl::half tanh(sycl::half x) {
  return sycl::tanh(x);
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CPP_
